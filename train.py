import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from transformers import get_cosine_schedule_with_warmup

from architectures import architecture_name, get_architecture
from dataset import TASK_TYPES, LengthGroupedBatchSampler, QuestionDataset, load_questions
from inference import autocast, evaluate, get_device, to_device
from logger import RunLogger, load_trainable_weights


def make_loader(split, config, arch, device, shuffle):
    data = config["data"]
    # max_test_questions caps whichever split is used for evaluation (data.eval_split)
    max_questions = data["max_train_questions"] if split == "train" else data["max_test_questions"]
    questions = load_questions(split, max_questions, config["seed"], data.get("task_types"))
    dataset = QuestionDataset(questions, arch.encode_example, data["max_state_tokens"], data["max_choice_tokens"])
    if shuffle and data.get("group_by_length", False):
        # Batches of similar state length (characters are a good enough proxy for tokens): less padding
        sampler = LengthGroupedBatchSampler([len(s) for s in questions["state"]], data["batch_size"], config["seed"])
        batching = {"batch_sampler": sampler}
    else:
        batching = {"batch_size": data["batch_size"], "shuffle": shuffle}
    return torch.utils.data.DataLoader(
        dataset,
        **batching,
        collate_fn=arch.collate_fn,
        num_workers=data["num_workers"],
        persistent_workers=data["num_workers"] > 0,  # keep workers alive across epochs and validation passes
        pin_memory=device.type == "cuda",
    )


def resolve_checkpoint(resume):
    """--resume takes a checkpoint folder, or an experiment id (runs/<id>/checkpoints/final)."""
    path = Path(resume)
    if not (path / "meta.json").exists():
        path = Path("runs") / resume / "checkpoints" / "final"
    if not (path / "meta.json").exists():
        raise FileNotFoundError(f"no checkpoint at {resume} or {path}")
    return path


SELECT_BY = ("accuracy", "loss")


def validate(network, test_loader, device, logger, step, best_score):
    """Evaluates and keeps checkpoints/best: highest accuracy, or lowest loss (NLL) with logging.select_by: loss.

    best_score is "higher is better": the accuracy, or minus the loss."""
    metrics = evaluate(network, test_loader, device)
    if device.type == "mps":
        torch.mps.empty_cache()  # give the evaluation's cached GPU memory back before training resumes
    logger.log(step, split=logger.config["data"].get("eval_split", "test"), **metrics)
    select_by = logger.config["logging"].get("select_by", "accuracy")
    score = metrics["accuracy"] if select_by == "accuracy" else -metrics["loss"]
    if score > best_score:
        best_score = score
        logger.save_checkpoint(network, name="best", step=step)
    return metrics, best_score


def train(config, run_name, resume=None, continue_run=None):
    """Trains one run from a config dict and returns its folder, runs/<run_name>-<timestamp>/.

    Saves checkpoints/best (best on data.eval_split, see logging.select_by), checkpoints/final and metrics.jsonl,
    and every logging.save_every steps checkpoints/latest, from which continue_run (a run folder) picks up
    exactly where the run stopped: same weights, optimizer, LR schedule, data position and best score.
    """
    torch.manual_seed(config["seed"])
    device = get_device()
    resume_dir, resume_meta, state = None, {}, None
    if continue_run:
        run_dir = Path(continue_run) if Path(continue_run).exists() else Path("runs") / continue_run
        config = yaml.safe_load((run_dir / "config.yaml").read_text())  # the run's own config, not the CLI one
        state = torch.load(run_dir / "checkpoints" / "latest" / "training_state.pt", map_location="cpu", weights_only=False)
        logger = RunLogger(config, None, device, run_dir=run_dir, elapsed=state["elapsed"])
    else:
        if resume:
            resume_dir = resolve_checkpoint(resume)
            resume_meta = json.loads((resume_dir / "meta.json").read_text())
            config = {**config, "resumed_from": str(resume_dir)}
        logger = RunLogger(config, run_name, device)
    model, optim, log = config["model"], config["optim"], config["logging"]
    select_by = log.get("select_by", "accuracy")
    if select_by not in SELECT_BY:
        raise ValueError(f"logging.select_by must be one of {SELECT_BY}, got {select_by!r}")

    arch = get_architecture(config)
    print(f"Architecture: {architecture_name(config)} ({model['name']})")
    network = arch.build_network(config, len(TASK_TYPES))
    if resume_dir:
        load_trainable_weights(network, resume_dir)
        print(f"Loaded trainable weights from {resume_dir} (step {resume_meta['step']})")
    if state:
        load_trainable_weights(network, logger.run_dir / "checkpoints" / "latest")
    network = network.to(device)
    count = lambda params: sum(p.numel() for p in params if p.requires_grad)
    print(f"trainable params: backbone={count(network.backbone.parameters()):,} head={count(network.head.parameters()):,}")

    train_loader = make_loader("train", config, arch, device, shuffle=True)
    test_loader = None  # evaluation loader; its split is data.eval_split ("test" for the original configs)
    if config["data"]["max_test_questions"] != 0:
        test_loader = make_loader(config["data"].get("eval_split", "test"), config, arch, device, shuffle=False)

    # Two learning rates: backbone (LoRA adapters or full fine-tuning) and everything else (head, answer query).
    # Weight decay only on weight matrices: biases and LayerNorm weights (1-d tensors) are not decayed.
    lora_params = [p for p in network.backbone.parameters() if p.requires_grad]
    head_params = [p for n, p in network.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    lrs = {"backbone": float(optim.get("lr_backbone", optim.get("lr_lora"))), "head": float(optim["lr_head"])}
    groups = []
    for name, params in [("backbone", lora_params), ("head", head_params)]:
        for decay in (True, False):
            selected = [p for p in params if (p.ndim >= 2) == decay]
            if selected:
                groups.append({"params": selected, "lr": lrs[name], "name": name,
                               "weight_decay": float(optim["weight_decay"]) if decay else 0.0})
    optimizer = torch.optim.AdamW(groups)
    group_of = {g["name"]: i for i, g in reversed(list(enumerate(groups)))}  # first group of each name, for logging
    # grad_accum_steps > 1 splits each optimizer step over that many loader batches (batch_size is per micro-batch)
    accum = optim.get("grad_accum_steps", 1)
    total_steps = optim["epochs"] * (len(train_loader) // accum)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(optim["warmup_ratio"] * total_steps), total_steps)

    print(f"{len(train_loader.dataset)} training questions -> {total_steps} optimizer steps")
    step, best_score, start_epoch, batches_done = 0, -float("inf"), 0, 0
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, best_score, start_epoch, batches_done = state["step"], state["best_score"], state["epoch"], state["batches_done"]
        torch.set_rng_state(state["rng"])
        sampler = train_loader.batch_sampler
        if batches_done and not isinstance(sampler, LengthGroupedBatchSampler):
            raise ValueError("continuing mid-epoch needs data.group_by_length: true (its batch order is reproducible)")
        if isinstance(sampler, LengthGroupedBatchSampler):
            sampler.epoch, sampler.skip = start_epoch, batches_done
        print(f"Continuing at step {step}/{total_steps} (epoch {start_epoch}, {batches_done} batches into it)")
    window_loss, window_correct, window_count = 0.0, 0, 0  # train metrics averaged since the last log line
    start, start_step = time.time(), step
    if resume_dir and test_loader is not None:
        _, best_score = validate(network, test_loader, device, logger, step, best_score)
    network.train()
    for epoch in range(start_epoch, optim["epochs"]):
        epoch_loss, epoch_correct, epoch_count = 0.0, 0, 0  # after --continue_run: only the part run since then
        optimizer.zero_grad()
        first = batches_done + 1 if epoch == start_epoch else 1
        for micro, batch in enumerate(train_loader, first):
            if micro > (len(train_loader) // accum) * accum:
                break  # drop the incomplete final accumulation group
            batch = to_device(batch, device)
            with autocast(device):
                logits = network(**batch)
            loss = F.cross_entropy(logits, batch["labels"])
            (loss / accum).backward()

            correct = (logits.argmax(-1) == batch["labels"]).sum().item()
            epoch_loss += loss.item() * len(logits)
            epoch_correct += correct
            epoch_count += len(logits)
            window_loss += loss.item() * len(logits)
            window_correct += correct
            window_count += len(logits)
            if micro % accum:
                continue

            torch.nn.utils.clip_grad_norm_(network.parameters(), optim["grad_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step == start_step + 20:
                per_step = (time.time() - start) / 20
                print(f"~{per_step:.1f} s/step -> about {per_step * (total_steps - step) / 60:.0f} min of training left (+ evaluations)")

            if step % log["log_every"] == 0:
                current = scheduler.get_last_lr()
                logged_lrs = {f"lr_{name}": current[i] for name, i in group_of.items()}
                logger.log(step, split="train", epoch=epoch, loss=window_loss / window_count,
                           accuracy=window_correct / window_count, **logged_lrs)
                window_loss, window_correct, window_count = 0.0, 0, 0
            if test_loader is not None and step % log["eval_every"] == 0:
                test_metrics, best_score = validate(network, test_loader, device, logger, step, best_score)
            if step % log["save_every"] == 0:  # after validation, so best_score is up to date
                logger.save_latest(network, {
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step,
                    "epoch": epoch, "batches_done": micro, "best_score": best_score, "rng": torch.get_rng_state(),
                    "elapsed": time.time() - logger.start_time}, step)

        epoch_count = max(epoch_count, 1)  # a run continued at the very end of an epoch has nothing left to train
        logger.log(step, split="train_epoch", epoch=epoch, loss=epoch_loss / epoch_count,
                   accuracy=epoch_correct / epoch_count)

    summary = {"final_train_loss": epoch_loss / epoch_count, "final_train_accuracy": epoch_correct / epoch_count}
    if test_loader is not None:
        if step % log["eval_every"] != 0:
            test_metrics, best_score = validate(network, test_loader, device, logger, step, best_score)
        summary["test"] = test_metrics
        summary[f"best_test_{select_by}"] = best_score if select_by == "accuracy" else -best_score

    ckpt_dir = logger.save_checkpoint(network, step=step)
    logger.finish(checkpoint=str(ckpt_dir), **summary)
    return logger.run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML config, e.g. configs/smoke.yaml")
    parser.add_argument("--name", help="run name, used as runs/<name>-<timestamp> (default: config file name)")
    parser.add_argument("--resume", help="start from a checkpoint's trainable weights: an experiment id "
                                         "(uses its final checkpoint) or a checkpoint folder. Optimizer and LR schedule start fresh.")
    parser.add_argument("--continue_run", help="continue an interrupted run in place, from runs/<id>/checkpoints/latest "
                                               "(weights, optimizer, LR schedule, data position). The config argument is then ignored.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    train(config, args.name or Path(args.config).stem, args.resume, args.continue_run)


if __name__ == "__main__":
    main()
