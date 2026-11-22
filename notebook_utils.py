"""Helpers shared by the notebooks: find runs, read their metrics, cache model outputs, plot calibration."""
import json
import time
from pathlib import Path

import torch
import yaml

from dataset import TASK_TYPES
from inference import collect_logits
from metrics import apply_temperature, reliability_bins

RUNS_ROOT = Path("runs")


def latest_run(name, root=RUNS_ROOT, finished=True):
    """Most recent runs/<name>-<timestamp>/ (that has a final checkpoint if finished=True), or None."""
    runs = sorted(root.glob(f"{name}-2*"), key=lambda p: p.name)  # timestamps sort chronologically
    if finished:
        runs = [r for r in runs if (r / "checkpoints" / "final" / "meta.json").exists()]
    return runs[-1] if runs else None


def read_metrics(run_dir):
    """metrics.jsonl grouped by split: {"train": [...], "validation": [...], ...}.

    Older train.py runs logged their periodic evaluation as "test" whatever data.eval_split was;
    those records are filed under the split that was actually evaluated."""
    eval_split = yaml.safe_load((Path(run_dir) / "config.yaml").read_text())["data"].get("eval_split", "test")
    history = {}
    for line in (Path(run_dir) / "metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        split = record.pop("split")
        history.setdefault(eval_split if split == "test" else split, []).append(record)
    return history


def cached_logits(network, ckpt_dir, split, loader, device):
    """collect_logits, saved next to the checkpoint so a rerun of the notebook does not recompute them.

    The cache is keyed by split and number of questions; it also stores the inference time per question.
    """
    path = Path(ckpt_dir) / f"logits_{split}_{len(loader.dataset)}.pt"
    if path.exists():
        return torch.load(path)
    start = time.time()
    logits, labels, task_types = collect_logits(network, loader, device)
    cache = {"logits": logits, "labels": labels, "task_types": task_types,
             "seconds_per_question": (time.time() - start) / len(labels)}
    torch.save(cache, path)
    return cache


def calibration_curves(logits, labels, task_types, temperature=1.0):
    """Per task, the (probability, outcome) pairs whose calibration matters:
    choice / score: top-choice confidence vs correct; noul: P(yes) vs the answer being yes."""
    probs = torch.softmax(apply_temperature(logits, task_types, temperature), -1)
    curves = {}
    for name, idx in TASK_TYPES.items():
        mask = task_types == idx
        if not mask.any():
            continue
        if name == "noul":
            curves[name] = (probs[mask, 1], labels[mask].float(), "P(yes)", "fraction yes")
        else:
            confidence, prediction = probs[mask].max(-1)
            curves[name] = (confidence, (prediction == labels[mask]).float(), "confidence", "accuracy")
    return curves


def plot_reliability(ax, probability, outcome, xlabel="predicted probability", ylabel="observed frequency",
                     label=None, n_bins=10, show_counts=True, **plot_kwargs):
    bins = reliability_bins(probability, outcome, n_bins)
    ax.plot([b["predicted"] for b in bins], [b["observed"] for b in bins], "o-", label=label, **plot_kwargs)
    if show_counts:
        for b in bins:
            ax.annotate(str(b["count"]), (b["predicted"], b["observed"]), fontsize=6, alpha=0.6,
                        textcoords="offset points", xytext=(0, 5), ha="center")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=xlabel, ylabel=ylabel)


def free_memory(device):
    import gc
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
