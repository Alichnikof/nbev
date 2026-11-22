import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from transformers import get_cosine_schedule_with_warmup

from architectures import architecture_name, get_architecture
from dataset import TASK_TYPES, QuestionDataset, load_questions
from inference import autocast, evaluate, get_device, to_device
from logger import RunLogger, load_trainable_weights


def make_loader(split, config, arch, device, shuffle):
    data = config["data"]
    # max_test_questions caps whichever split is used for evaluation (data.eval_split)
    max_questions = data["max_train_questions"] if split == "train" else data["max_test_questions"]
    questions = load_questions(split, max_questions, config["seed"])
    dataset = QuestionDataset(questions, arch.encode_example, data["max_state_tokens"], data["max_choice_tokens"])
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=data["batch_size"],
        shuffle=shuffle,
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


def validate(network, test_loader, device, logger, step, best_accuracy):
    metrics = evaluate(network, test_loader, device)
    logger.log(step, split=logger.config["data"].get("eval_split", "test"), **metrics)
    if metrics["accuracy"] > best_accuracy:
        best_accuracy = metrics["accuracy"]
        logger.save_checkpoint(network, name="best", step=step)
    return metrics, best_accuracy


def train(config, run_name, resume=None):
    torch.manual_seed(config["seed"])
    device = get_device()
    resume_dir, resume_meta = None, {}
    if resume:
        resume_dir = resolve_checkpoint(resume)
        resume_meta = json.loads((resume_dir / "meta.json").read_text())
        config = {**config, "resumed_from": str(resume_dir)}
    logger = RunLogger(config, run_name, device)
    model, optim, log = config["model"], config["optim"], config["logging"]

    arch = get_architecture(config)
    print(f"Architecture: {architecture_name(config)} ({model['name']})")
    network = arch.build_network(config, len(TASK_TYPES))
    if resume_dir:
        load_trainable_weights(network, resume_dir)
        print(f"Loaded trainable weights from {resume_dir} (step {resume_meta['step']})")
    network = network.to(device)
    count = lambda params: sum(p.numel() for p in params if p.requires_grad)
    print(f"trainable params: backbone={count(network.backbone.parameters()):,} head={count(network.head.parameters()):,}")

    train_loader = make_loader("train", config, arch, device, shuffle=True)
    test_loader = None  # evaluation loader; its split is data.eval_split ("test" for the original configs)
    if config["data"]["max_test_questions"] != 0:
        test_loader = make_loader(config["data"].get("eval_split", "test"), config, arch, device, shuffle=False)

    # Two learning rates: backbone (LoRA adapters or full fine-tuning) and everything else (head, answer query)
    lora_params = [p for p in network.backbone.parameters() if p.requires_grad]
    head_params = [p for n, p in network.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": float(optim.get("lr_backbone", optim.get("lr_lora")))},
        {"params": head_params, "lr": float(optim["lr_head"])},
    ], weight_decay=float(optim["weight_decay"]))
    # grad_accum_steps > 1 splits each optimizer step over that many loader batches (batch_size is per micro-batch)
    accum = optim.get("grad_accum_steps", 1)
    total_steps = optim["epochs"] * (len(train_loader) // accum)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(optim["warmup_ratio"] * total_steps), total_steps)

    step = 0
    best_accuracy = -1.0
    if resume_dir and test_loader is not None:
        _, best_accuracy = validate(network, test_loader, device, logger, step, best_accuracy)
    network.train()
    for epoch in range(optim["epochs"]):
        epoch_loss, epoch_correct, epoch_count = 0.0, 0, 0
        optimizer.zero_grad()
        for micro, batch in enumerate(train_loader, 1):
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
            if micro % accum:
                continue

            torch.nn.utils.clip_grad_norm_(network.parameters(), optim["grad_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % log["log_every"] == 0:
                lrs = {"lr_head": scheduler.get_last_lr()[1]}
                if lora_params:
                    lrs["lr_backbone"] = scheduler.get_last_lr()[0]
                logger.log(step, split="train", epoch=epoch, loss=loss.item(), accuracy=correct / len(logits), **lrs)
            if step % log["save_every"] == 0:
                logger.save_checkpoint(network, name="latest", step=step)
            if test_loader is not None and step % log["eval_every"] == 0:
                test_metrics, best_accuracy = validate(network, test_loader, device, logger, step, best_accuracy)

        logger.log(step, split="train_epoch", epoch=epoch, loss=epoch_loss / epoch_count,
                   accuracy=epoch_correct / epoch_count)

    summary = {"final_train_loss": epoch_loss / epoch_count, "final_train_accuracy": epoch_correct / epoch_count}
    if test_loader is not None:
        if step % log["eval_every"] != 0:
            test_metrics, best_accuracy = validate(network, test_loader, device, logger, step, best_accuracy)
        summary["test"] = test_metrics
        summary["best_test_accuracy"] = best_accuracy

    ckpt_dir = logger.save_checkpoint(network, step=step)
    logger.finish(checkpoint=str(ckpt_dir), **summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML config, e.g. configs/smoke.yaml")
    parser.add_argument("--name", help="run name, used as runs/<name>-<timestamp> (default: config file name)")
    parser.add_argument("--resume", help="start from a checkpoint's trainable weights: an experiment id "
                                         "(uses its final checkpoint) or a checkpoint folder. Optimizer and LR schedule start fresh.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    train(config, args.name or Path(args.config).stem, args.resume)


if __name__ == "__main__":
    main()
