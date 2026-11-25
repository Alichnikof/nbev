"""Helpers for the notebooks, so that each notebook cell is one readable step.

Nothing here contains model logic: it calls train.py, logger.py, calibration.py, metrics.py and
inference.py, and adds plotting, printing and caching on top.

    Runs         train_or_load, latest_run, read_metrics
    Data         show_question, show_model_input, plot_attention_mask
    Evaluation   evaluate_split (cached), free_memory
    Inspection   parameter_table, layer_summary, head_attention, plot_head_attention
    Plots        plot_training_curves, plot_calibration, plot_test_results
    Analysis     baseline_table, plot_noul_threshold, plot_score_errors, accuracy_by, show_predictions, inference_speed
    Hands-on     ask
"""
import gc
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from architectures import get_architecture
from dataset import TASK_NAMES, TASK_TYPES, QuestionDataset, load_questions
from inference import answer, collect_logits, to_device
from metrics import apply_temperature, reliability_bins

RUNS_ROOT = Path("runs")


# ----------------------------------------------------------------------------------------- runs

def latest_run(name, root=RUNS_ROOT):
    """The most recent finished run folder runs/<name>-<timestamp>/ (it has checkpoints/final), or None."""
    runs = sorted(root.glob(f"{name}-2*"), key=lambda p: p.name)  # timestamps sort chronologically
    runs = [r for r in runs if (r / "checkpoints" / "final" / "meta.json").exists()]
    return runs[-1] if runs else None


def train_or_load(config, run_name, retrain=False):
    """Returns the folder of a finished run called run_name, training one with train.py only if needed.

    retrain=False and a finished run exists -> nothing is trained or overwritten, that run is reused.
    Otherwise                                -> train.train(config, run_name) creates a new timestamped run.
    """
    run_dir = None if retrain else latest_run(run_name)
    if run_dir is not None:
        print(f"Reusing finished run {run_dir} (nothing is retrained; pass retrain=True to train a new one).")
        return run_dir
    from train import train
    run_dir = train(config, run_name)
    free_memory()
    return run_dir


def read_metrics(run_dir):
    """metrics.jsonl grouped by split: {"train": [...], "validation": [...], "train_epoch": [...]}.

    Older train.py runs logged their periodic evaluation as "test" whatever data.eval_split was;
    those records are filed under the split that was actually evaluated."""
    eval_split = yaml.safe_load((Path(run_dir) / "config.yaml").read_text())["data"].get("eval_split", "test")
    history = {}
    for line in (Path(run_dir) / "metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        split = record.pop("split")
        history.setdefault(eval_split if split == "test" else split, []).append(record)
    return history


# ----------------------------------------------------------------------------------------- data

def show_question(row, state_chars=300):
    """Prints one dataset question: type, instructions, start of the state, candidates (→ = correct)."""
    print(f"[{TASK_NAMES[row['task_type']]}] {row['instructions']}")
    print("   state:", row["state"][:state_chars].replace("\n", " ") + (" …" if len(row["state"]) > state_chars else ""))
    for i, choice in enumerate(row["choices"]):
        print(f"   {'→' if i == row['label'] else ' '} {i}. {choice[:110]}")
    print()


def show_model_input(arch, row, data_config, max_context_chars=1500):
    """Prints one question exactly as nbev reads it: the decoded packed sequence, block by block.

    The context (task hint + question + state) is decoded from the real token ids, so a long state shows the
    model's own head + "..." + tail truncation. Only for display, a context longer than max_context_chars
    characters is shortened in the middle; the model itself reads every token of it."""
    tokenizer = arch.tokenization.tokenizer
    encoded = arch.encode_example(row["state"], row["instructions"], row["choices"], TASK_NAMES[row["task_type"]],
                                  data_config["max_state_tokens"], data_config["max_choice_tokens"])
    ids, positions, n = encoded["input_ids"], encoded["position_ids"], encoded["context_len"]
    context = tokenizer.decode(ids[:n])
    if len(context) > max_context_chars:
        keep = max_context_chars // 2
        context = (context[:keep] + f"\n      [... {len(context) - 2 * keep} characters not printed here, "
                   f"but read by the model ...]\n" + context[-keep:])
    print(f"=== {TASK_NAMES[row['task_type']]} | {len(ids)} tokens: context = positions 0..{n - 1}, "
          f"{len(encoded['blocks'])} candidate blocks ===")
    print(context)
    for i, (start, end) in enumerate(encoded["blocks"]):
        mark = "  <- correct answer" if i == row["label"] else ""
        print(f"[OPT] {tokenizer.decode(ids[start + 1:end - 1])} [SEP]   (positions {positions[start]}..{positions[end - 1]}){mark}")
    print()


MASK_COLORS = ["#9aa5b1", "#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6", "#a16207",
               "#6366f1", "#84cc16"]  # grey = context, then one colour per option


def _block_name(tokenizer, ids, start, end, chars=18):
    """Short name of an option block: its decoded text up to the first ":" (dataset options are "key: description")."""
    text = tokenizer.decode(ids[start + 1:end - 1]).split(":")[0].strip()
    return text if len(text) <= chars else text[:chars - 1] + "…"


def plot_attention_mask(arch, examples, titles, max_labelled_tokens=60):
    """Draws the attention mask of encoded questions (outputs of arch.encode_example), side by side, and returns a
    table that checks the masking rules on each of them.

    Row = the token doing the looking (query), column = the token being looked at (key). An allowed cell takes the
    colour of the key's owner (grey = context, one colour per option); white = blocked. A sequence of at most
    max_labelled_tokens gets one label per token, "token (position id)"; a longer one gets one label per block."""
    from matplotlib.colors import ListedColormap
    tokenization = arch.tokenization
    tokenizer = tokenization.tokenizer
    fig, axes = plt.subplots(1, len(examples), figsize=(8.5 * len(examples), 8.8), squeeze=False)
    checks = {}
    for ax, example, title in zip(axes[0], examples, titles):
        ids, positions, n, blocks = example["input_ids"], example["position_ids"], example["context_len"], example["blocks"]
        L = len(ids)
        mask = tokenization.attention_mask_for(example, L)
        owner = np.zeros(L, dtype=int)  # 0 = context, i = option i
        for i, (start, end) in enumerate(blocks, 1):
            owner[start:end] = i
        colors = [MASK_COLORS[0]] + [MASK_COLORS[1 + i % (len(MASK_COLORS) - 1)] for i in range(len(blocks))]
        grid = np.where(mask.numpy(), owner[None, :] + 1, 0)
        ax.imshow(grid, cmap=ListedColormap(["white", *colors]), vmin=-0.5, vmax=len(colors) + 0.5,
                  interpolation="nearest")
        if L <= max_labelled_tokens:
            token = lambda t: "[OPT]" if t == tokenization.OPT_TOKEN_ID else tokenizer.convert_ids_to_tokens(t).replace("▁", "")
            ticks, labels = list(range(L)), [f"{token(t)}  ({p})" for t, p in zip(ids, positions)]
            tick_owner = owner
            ax.set_xticks(np.arange(L + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(L + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="#e4e7eb", lw=0.5)
            ax.tick_params(which="minor", length=0)
        else:
            ticks = [(n - 1) / 2] + [(start + end - 1) / 2 for start, end in blocks]
            labels = [f"context (positions 0..{n - 1})"] + [
                f"{_block_name(tokenizer, ids, start, end)} ({positions[start]}..{positions[end - 1]})" for start, end in blocks]
            tick_owner = range(len(blocks) + 1)
            for boundary in [n] + [start for start, _ in blocks[1:]]:
                ax.axhline(boundary - 0.5, color="#cbd2d9", lw=0.6)
                ax.axvline(boundary - 0.5, color="#cbd2d9", lw=0.6)
        fontsize = 7 if L <= max_labelled_tokens else 9
        ax.set_xticks(ticks, labels, rotation=90, fontsize=fontsize)
        ax.set_yticks(ticks, labels, fontsize=fontsize)
        for tick_labels in (ax.get_xticklabels(), ax.get_yticklabels()):
            for label, o in zip(tick_labels, tick_owner):
                label.set_color(colors[o] if o else "#3e4c59")
        ax.set_xlabel("key: the token being looked at")
        ax.set_ylabel("query: the token doing the looking")
        ax.set_title(title, fontsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)

        others = [mask[a0:a1, b0:b1].any() for i, (a0, a1) in enumerate(blocks) for j, (b0, b1) in enumerate(blocks) if i != j]
        checks[title.split(":")[0]] = {
            "tokens": L, "options": len(blocks),
            "context sees an option": bool(mask[:n, n:].any()),
            "an option sees another option": bool(any(others)),
            "every option sees the whole context": all(bool(mask[a0:a1, :n].all()) for a0, a1 in blocks),
            "every option starts at position": ", ".join(sorted({str(positions[a0]) for a0, _ in blocks})),
        }
    plt.tight_layout()
    plt.show()
    return pd.DataFrame(checks).T


# ----------------------------------------------------------------------------------------- evaluation

def free_memory():
    """Releases cached GPU memory, e.g. after deleting a model."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_split(network, config, ckpt_dir, split, max_questions, device, batch_size=8):
    """Runs the model on the first max_questions of a split and returns its raw outputs:
    {"logits" [N, C], "labels" [N], "task_types" [N], "seconds_per_question"}.

    The result is cached as ckpt_dir/logits_<split>_<N>.pt, so running the notebook again does not
    recompute it. Metrics are computed from these outputs with metrics.compute_metrics.
    """
    task_types = config["data"].get("task_types")
    questions = load_questions(split, max_questions, config["seed"], task_types)
    suffix = "" if task_types is None else "_" + "-".join(task_types)
    path = Path(ckpt_dir) / f"logits_{split}{suffix}_{len(questions)}.pt"
    if path.exists():
        return torch.load(path)
    arch, d = get_architecture(config), config["data"]
    dataset = QuestionDataset(questions, arch.encode_example, d["max_state_tokens"], d["max_choice_tokens"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=arch.collate_fn)
    start = time.time()
    logits, labels, task_types = collect_logits(network, loader, device)
    outputs = {"logits": logits, "labels": labels, "task_types": task_types,
               "seconds_per_question": (time.time() - start) / len(labels)}
    torch.save(outputs, path)
    return outputs


# ----------------------------------------------------------------------------------------- inspection

def parameter_table(network):
    """Parameter count of each top-level block of the backbone and of the head."""
    rows = [(f"backbone.{name}", module) for name, module in network.backbone.named_children()]
    rows += [(f"head.{name}", module) for name, module in network.head.named_children()]
    table = pd.DataFrame([{"block": name, "type": type(m).__name__, "parameters": sum(p.numel() for p in m.parameters()),
                           "trainable": sum(p.numel() for p in m.parameters() if p.requires_grad)} for name, m in rows])
    table = table[table.parameters > 0]
    total = {"block": "total", "type": "", "parameters": table.parameters.sum(), "trainable": table.trainable.sum()}
    return pd.concat([table, pd.DataFrame([total])], ignore_index=True).style.format("{:,}", subset=["parameters", "trainable"])


def layer_summary(network, batch, device, depth=4):
    """torchinfo table (module, output shape, parameters) from one real forward pass.
    DeBERTa's 12 encoder layers are identical, so only the first one is printed in full."""
    from torchinfo import summary
    stats = summary(network, input_data=dict(to_device(batch, device)), depth=depth, verbose=0,
                    col_names=("output_size", "num_params"), row_settings=("var_names",))
    lines = str(stats).splitlines()
    starts = [i for i, l in enumerate(lines) if "(1)" in l]
    ends = [i for i, l in enumerate(lines) if "(11)" in l]
    if starts and ends:
        lines = lines[:starts[0]] + ["│    │    │    └─ … layers (1) to (11): same as (0) …"] + lines[ends[-1] + 1:]
    print("\n".join(lines))


@torch.no_grad()
def head_attention(network, config, device, state, instructions, task_type, criteria=None):
    """Runs one question and returns the decision head's attention between candidates, one [C, C] array per
    head layer (row = the candidate that looks, column = the candidate it looks at; each row sums to 1).
    The weights are recomputed from the inputs of each SelfAttention with its own q and k projections."""
    from dataset import question_to_choices
    from inference import autocast
    question = {"type": task_type, "instructions": instructions}
    if criteria is not None:
        question["criteria"] = criteria
    choices, _ = question_to_choices(question)
    arch, d = get_architecture(config), config["data"]
    example = arch.encode_example(state, instructions, choices, task_type, d["max_state_tokens"], d["max_choice_tokens"])
    example.update(task_type=TASK_TYPES[task_type], label=None)
    batch = to_device(arch.collate_fn([example]), device)

    captured = []
    def hook(module, inputs, output):
        x, mask = inputs
        scores = module.q(x) @ module.k(x).transpose(-1, -2) / module.new_dim ** 0.5
        captured.append(torch.softmax(scores.masked_fill(~mask[:, None, :], -float("inf")), -1)[0].float().cpu().numpy())
    handles = [layer.attn.register_forward_hook(hook) for layer in network.head.layers]
    network.eval()
    try:
        with autocast(device):
            network(**batch)
    finally:
        for h in handles:
            h.remove()
    return captured, choices


def plot_head_attention(weights, choices):
    """One heatmap per head layer from head_attention()."""
    labels = [c if len(c) <= 25 else c[:24] + "…" for c in choices]
    fig, axes = plt.subplots(1, len(weights), figsize=(4.6 * len(weights), 4), squeeze=False)
    for i, (ax, w) in enumerate(zip(axes[0], weights)):
        ax.imshow(w, cmap="Blues", vmin=0, vmax=1)
        for r in range(len(w)):
            for c in range(len(w)):
                ax.text(c, r, f"{w[r, c]:.2f}", ha="center", va="center", color="white" if w[r, c] > 0.5 else "black")
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set(title=f"head layer {i}: who looks at whom", xlabel="looked at (key)", ylabel="looking (query)")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------------------- plots

def plot_training_curves(run_dir):
    """Train and validation loss / accuracy from metrics.jsonl; the dashed line is the best checkpoint."""
    history = read_metrics(run_dir)
    train, val = history["train"], history["validation"]
    best_step = json.loads((Path(run_dir) / "checkpoints" / "best" / "meta.json").read_text())["step"]
    tasks = [t for t in TASK_TYPES if any(f"accuracy_{t}" in r for r in val)]
    per_task = len(tasks) > 1  # with a single task type, its accuracy is the overall one: no third panel
    fig, axes = plt.subplots(1, 3 if per_task else 2, figsize=(16, 4) if per_task else (11, 4))
    for ax, key in [(axes[0], "loss"), (axes[1], "accuracy")]:
        ax.plot([r["step"] for r in train], [r[key] for r in train], label="train", alpha=0.8)
        ax.plot([r["step"] for r in val], [r[key] for r in val], "o-", label="validation")
        ax.set_title(key)
    if per_task:
        for task in tasks:
            axes[2].plot(*zip(*[(r["step"], r[f"accuracy_{task}"]) for r in val if f"accuracy_{task}" in r]), "o-", label=task)
        axes[2].set_title("validation accuracy per task type")
    for ax in axes:
        ax.axvline(best_step, ls="--", c="grey", lw=1)
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.show()


def _calibration_points(outputs, temperature):
    """Per task, the (probability, outcome) pairs whose calibration matters:
    noul: P(yes) vs "the answer is yes"; choice and score: top-candidate confidence vs "it is correct"."""
    probs = torch.softmax(apply_temperature(outputs["logits"], outputs["task_types"], temperature), -1)
    labels, points = outputs["labels"], {}
    for name, idx in TASK_TYPES.items():
        mask = outputs["task_types"] == idx
        if not mask.any():
            continue
        if name == "noul":
            points[name] = (probs[mask, 1], labels[mask].float(), "predicted P(yes)", "observed fraction of yes")
        else:
            confidence, prediction = probs[mask].max(-1)
            points[name] = (confidence, (prediction == labels[mask]).float(), "confidence", "accuracy")
    return points


def _reliability_curve(ax, probability, outcome, **kwargs):
    bins = reliability_bins(probability, outcome)
    ax.plot([b["predicted"] for b in bins], [b["observed"] for b in bins], "o-", **kwargs)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlim=(0, 1), ylim=(0, 1))


def plot_calibration(outputs, temperatures, title_suffix=""):
    """One reliability diagram per task type, before (T = 1) and after temperature scaling.
    On the diagonal = calibrated; below = over-confident; above = under-confident."""
    before, after = _calibration_points(outputs, 1.0), _calibration_points(outputs, temperatures)
    fig, axes = plt.subplots(1, len(before), figsize=(5.3 * len(before), 4.5), squeeze=False)  # 2-d even for one task
    for ax, task in zip(axes[0], before):
        p, y, xlabel, ylabel = before[task]
        _reliability_curve(ax, p, y, label="T = 1", alpha=0.5)
        _reliability_curve(ax, *after[task][:2], label=f"T = {temperatures[task]:.2f}")
        ax.set(title=f"{task} ({len(p)} questions){title_suffix}", xlabel=xlabel, ylabel=ylabel)
        ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_test_results(outputs, metrics, temperatures):
    """Left: test accuracy per task type. Right: calibrated reliability per task type."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    tasks = [t for t in TASK_TYPES if f"accuracy_{t}" in metrics]
    accuracies = [metrics[f"accuracy_{t}"] for t in tasks]
    axes[0].bar(tasks, accuracies, color=["C0", "C1", "C2"][:len(tasks)])
    for i, a in enumerate(accuracies):
        axes[0].text(i, a + 0.01, f"{a:.3f}", ha="center")
    axes[0].axhline(metrics["accuracy"], ls="--", c="grey", label=f"overall {metrics['accuracy']:.3f}")
    axes[0].set(ylim=(0, 1), title="test accuracy per task type")
    axes[0].legend()
    for task, (p, y, *_ ) in _calibration_points(outputs, temperatures).items():
        _reliability_curve(axes[1], p, y, label=task)
    axes[1].set(title="test reliability, calibrated (noul: P(yes); others: confidence)",
                xlabel="predicted probability", ylabel="observed frequency")
    axes[1].legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------------------- analysis

def _probs(outputs, temperatures):
    return torch.softmax(apply_temperature(outputs["logits"], outputs["task_types"], temperatures), -1)


def _num_candidates(outputs):
    return torch.isfinite(outputs["logits"]).sum(-1)


def baseline_table(outputs, temperatures):
    """Accuracy of the model next to what trivial strategies would get, per task type.

    chance        pick a candidate at random: the mean of 1 / (number of candidates)
    majority      always give the most frequent answer: "No" for noul; for score, the most frequent level of
                  each scale size (counted on this same split, so it flatters the baseline)
    For score, the ordinal quality is also shown: within-1 accuracy and MAE of the expected score,
    next to the MAE of always predicting the middle of the scale."""
    probs, labels, tasks, n_cand = _probs(outputs, temperatures), outputs["labels"], outputs["task_types"], _num_candidates(outputs)
    prediction = probs.argmax(-1)
    rows = []
    for name, idx in TASK_TYPES.items():
        m = tasks == idx
        if not m.any():
            continue
        y, pred, c = labels[m], prediction[m], n_cand[m]
        row = {"task": name, "questions": int(m.sum()), "model accuracy": (pred == y).float().mean().item(),
               "chance": (1 / c.float()).mean().item()}
        if name == "noul":
            row["majority"] = max((y == 0).float().mean().item(), (y == 1).float().mean().item())
        elif name == "score":
            row["majority"] = sum(torch.bincount(y[c == k]).max().item() for k in c.unique()) / len(y)
            expected = (probs[m] * torch.arange(probs.shape[1])).sum(-1)
            row["within-1 accuracy"] = ((pred - y).abs() <= 1).float().mean().item()
            row["MAE"] = (expected - y).abs().mean().item()
            row["MAE of 'middle level'"] = ((c - 1) / 2 - y).abs().mean().item()
        rows.append(row)
    return pd.DataFrame(rows).set_index("task")


def _binary_scores(p_yes, is_yes, threshold):
    pred = p_yes >= threshold
    tp, fp = (pred & is_yes).sum().item(), (pred & ~is_yes).sum().item()
    fn, tn = (~pred & is_yes).sum().item(), (~pred & ~is_yes).sum().item()
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"threshold": threshold, "accuracy": (tp + tn) / len(pred),
            "balanced accuracy": (recall + tn / max(tn + fp, 1)) / 2,
            "precision (yes)": precision, "recall (yes)": recall,
            "F1 (yes)": 2 * precision * recall / max(precision + recall, 1e-9), "predicted yes": pred.float().mean().item()}


def plot_noul_threshold(calibration, test, temperatures):
    """Yes/no decisions: how P(yes) separates the answers, and what a threshold other than 0.5 would do.

    The accuracy reported everywhere else is the argmax over ["No", "Yes"], i.e. the threshold P(yes) >= 0.5.
    Here thresholds are chosen on the CALIBRATION split (best accuracy, best F1 of "yes") and then applied,
    unchanged, to the test split. Returns the table of test scores for each threshold."""
    def p_and_y(outputs):
        m = outputs["task_types"] == TASK_TYPES["noul"]
        return _probs(outputs, temperatures)[m, 1], outputs["labels"][m].bool()
    p_cal, y_cal = p_and_y(calibration)
    p_test, y_test = p_and_y(test)
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    sweep = pd.DataFrame([_binary_scores(p_cal, y_cal, t) for t in grid])
    chosen = {"argmax (0.5)": 0.5,
              "best accuracy on calibration": float(sweep.loc[sweep["accuracy"].idxmax(), "threshold"]),
              "best F1 (yes) on calibration": float(sweep.loc[sweep["F1 (yes)"].idxmax(), "threshold"])}

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    bins = np.linspace(0, 1, 26)
    axes[0].hist(p_test[~y_test].numpy(), bins, alpha=0.6, label=f"true answer No ({int((~y_test).sum())})")
    axes[0].hist(p_test[y_test].numpy(), bins, alpha=0.6, label=f"true answer Yes ({int(y_test.sum())})")
    axes[0].set(title="test: predicted P(yes), split by the true answer", xlabel="calibrated P(yes)", ylabel="questions")
    for metric in ["accuracy", "balanced accuracy", "F1 (yes)"]:
        axes[1].plot(sweep["threshold"], sweep[metric], label=metric)
    axes[1].axhline(max(y_cal.float().mean().item(), 1 - y_cal.float().mean().item()), c="grey", ls=":", label="always No")
    axes[1].set(title="calibration split: score of the rule P(yes) >= threshold", xlabel="threshold")
    for ax in axes:
        for (name, t), color in zip(chosen.items(), ["k", "C3", "C4"]):
            ax.axvline(t, c=color, ls="--", lw=1)
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return pd.DataFrame({name: _binary_scores(p_test, y_test, t) for name, t in chosen.items()}).T


def plot_score_errors(outputs, temperatures):
    """Score questions: confusion matrix of the most probable level, and the distribution of (predicted - true)."""
    m = outputs["task_types"] == TASK_TYPES["score"]
    probs, y = _probs(outputs, temperatures)[m], outputs["labels"][m]
    pred = probs.argmax(-1)
    K = int(max(y.max(), pred.max())) + 1
    confusion = np.zeros((K, K), dtype=int)
    for t, p in zip(y.tolist(), pred.tolist()):
        confusion[t, p] += 1
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].imshow(confusion / confusion.sum(1, keepdims=True).clip(min=1), cmap="Blues", vmin=0, vmax=1)
    for t in range(K):
        for p in range(K):
            if confusion[t, p]:
                axes[0].text(p, t, confusion[t, p], ha="center", va="center")
    axes[0].set(title="score: true level (rows) vs most probable level (columns)\ncolour = share of the row",
                xlabel="predicted level", ylabel="true level", xticks=range(K), yticks=range(K))
    diff = (pred - y).numpy()
    values, counts = np.unique(diff, return_counts=True)
    axes[1].bar(values, counts / len(diff))
    axes[1].set(title=f"predicted - true level  (exact {np.mean(diff == 0):.2f}, within 1 {np.mean(np.abs(diff) <= 1):.2f})",
                xlabel="error in levels", ylabel="share of score questions")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def accuracy_by(outputs, questions, key, temperatures, min_questions=5):
    """Accuracy grouped by a question field ("domain" or "instructions"), with the chance level of each group.
    questions must be the same split and size as outputs (as returned by load_questions), in the same order."""
    hits = (_probs(outputs, temperatures).argmax(-1) == outputs["labels"]).float().numpy()
    table = pd.DataFrame({key: questions[key], "task": [TASK_NAMES[t] for t in questions["task_type"]],
                          "hit": hits, "chance": (1 / _num_candidates(outputs).float()).numpy()})
    if key == "instructions":
        table[key] = table[key].str.slice(0, 90)
    grouped = table.groupby([key, "task"]).agg(questions=("hit", "size"), accuracy=("hit", "mean"), chance=("chance", "mean"))
    grouped = grouped[grouped.questions >= min_questions]
    return grouped.assign(gain=grouped.accuracy - grouped.chance).sort_values("questions", ascending=False)


def show_predictions(outputs, questions, temperatures, n=4, mistakes=False, successes=False, min_confidence=0.7,
                     seed=0, state_chars=300):
    """Prints n questions with the model's calibrated probabilities as bars (→ = correct answer).

    Default: n random questions. mistakes=True: the n wrong answers the model was most confident about.
    successes=True: n right answers given with at least min_confidence, one per domain (largest domains first),
    picked at random within the domain, after a line saying how often the model is that confident and right.
    questions must be the same split and size as outputs (as returned by load_questions), in the same order.
    A state longer than state_chars is printed as its start + " … " + its end."""
    probs = _probs(outputs, temperatures)
    confidence, prediction = probs.max(-1)
    wrong = prediction != outputs["labels"]
    order = torch.randperm(len(probs), generator=torch.Generator().manual_seed(seed)).tolist()
    if mistakes:
        indices = [i for i in confidence.argsort(descending=True).tolist() if wrong[i]][:n]
    elif successes:
        confident = confidence >= min_confidence
        print(f"{confident.float().mean():.0%} of the {len(probs)} questions are answered with confidence ≥ "
              f"{min_confidence}, and {(~wrong[confident]).float().mean():.0%} of those answers are right.\n")
        domains = pd.Series(questions["domain"])
        first_right = {}  # domain -> a random confident right answer in it
        for i in order:
            if confident[i] and not wrong[i]:
                first_right.setdefault(domains[i], i)
        indices = [first_right[domain] for domain in domains.value_counts().index if domain in first_right][:n]
    else:
        indices = order[:n]
    for i in indices:
        row = questions[i]
        state = " ".join(row["state"].split())
        if len(state) > state_chars:
            head = 2 * state_chars // 3
            state = state[:head] + " … " + state[-(state_chars - head):]
        print(f"[{row['domain']}] {row['instructions']}")
        print(f"   {'✗ wrong' if wrong[i] else '✓ right'}, confidence {confidence[i]:.2f}")
        print(f"   state: {state}")
        for j, choice in enumerate(row["choices"]):
            print(f"   {'→' if j == row['label'] else ' '} {probs[i, j]:.3f}  {'█' * round(probs[i, j].item() * 30):<30}  {choice[:100]}")
        print()


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def inference_speed(network, config, device, questions, batch_sizes=(1, 8), num_questions=64,
                    candidate_counts=(2, 4, 8, 16, 32), repeats=5):
    """End-to-end inference time (tokenization + collate + one forward pass + softmax). Returns two tables:

    by batch size        ms per question and questions per second on num_questions real questions, batched by length
    by candidate count   ms for one question (batch 1) with 2 ... 32 candidates on the same state: one forward
                         pass whatever the count, the context is encoded once and each candidate adds a short block
    Every timed pass is first run once untimed: MPS / CUDA compile kernels for each new input shape.
    """
    from inference import autocast
    arch, d = get_architecture(config), config["data"]
    network.eval()

    def run(rows):
        examples = []
        for r in rows:
            ex = arch.encode_example(r["state"], r["instructions"], r["choices"], TASK_NAMES[r["task_type"]],
                                     d["max_state_tokens"], d["max_choice_tokens"])
            ex.update(task_type=r["task_type"], label=r.get("label"))
            examples.append(ex)
        with autocast(device):
            torch.softmax(network(**to_device(arch.collate_fn(examples), device)).float(), -1).cpu()
        return sum(len(ex["input_ids"]) for ex in examples)

    # Batches of similar length, as in training: a batch is padded to its longest question
    rows = sorted((questions[i] for i in range(min(num_questions, len(questions)))), key=lambda r: len(r["state"]))
    by_batch = []
    for bs in batch_sizes:
        batches = [rows[i:i + bs] for i in range(0, len(rows), bs)]
        tokens = sum(run(batch) for batch in batches)     # warm-up on every shape
        _sync(device)
        start = time.perf_counter()
        for batch in batches:
            run(batch)
        _sync(device)
        seconds = (time.perf_counter() - start) / len(rows)
        by_batch.append({"batch size": bs, "ms per question": 1000 * seconds, "questions per second": 1 / seconds,
                         "mean tokens per question": tokens / len(rows)})

    state = rows[len(rows) // 2]["state"]                 # a state of median length
    by_count = []
    for count in candidate_counts:
        row = {"state": state, "instructions": "Which option fits best?", "task_type": TASK_TYPES["choice"],
               "choices": [f"option {i}: a short description of choice number {i}" for i in range(count)]}
        tokens = run([row])                               # warm-up
        times = []
        for _ in range(repeats):
            _sync(device)
            start = time.perf_counter()
            run([row])
            _sync(device)
            times.append(time.perf_counter() - start)
        by_count.append({"candidates": count, "tokens": tokens, "ms per question": 1000 * float(np.median(times))})
    return pd.DataFrame(by_batch).set_index("batch size"), pd.DataFrame(by_count).set_index("candidates")


# ----------------------------------------------------------------------------------------- hands-on

def ask(network, config, device, state, instructions, task_type, criteria=None):
    """Asks the model one question and prints the calibrated answer as bars. Returns answer()'s dict.

    task_type "choice": criteria = {"key": "description", ...}   (descriptions may be "")
    task_type "score":  criteria = ["lowest level", ..., "highest level"]
    task_type "noul":   no criteria, the answer is P(yes)
    """
    question = {"type": task_type, "instructions": instructions}
    if criteria is not None:
        question["criteria"] = criteria
    result = answer(network, config, state, question, device)

    print(f"[{task_type}] {instructions}")
    if task_type == "noul":
        print(f"   P(yes) = {result['noul']:.3f}  {'█' * round(result['noul'] * 30)}")
    else:
        names = list(criteria) if task_type == "choice" else [f"{i}: {c}" for i, c in enumerate(criteria)]
        probs = result["probabilities"]
        probs = [probs[k] for k in criteria] if isinstance(probs, dict) else probs
        for name, p in zip(names, probs):
            print(f"   {p:6.3f}  {'█' * round(p * 30):<30}  {name}")
        if task_type == "score":
            print(f"   expected score = {result['score']:.2f}")
    print()
    return result
