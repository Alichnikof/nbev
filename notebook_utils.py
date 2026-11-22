"""Helpers for the notebooks, so that each notebook cell is one readable step.

Nothing here contains model logic: it calls train.py, logger.py, calibration.py, metrics.py and
inference.py, and adds plotting, printing and caching on top.

    Runs         train_or_load, latest_run, read_metrics
    Data         show_question
    Evaluation   evaluate_split (cached), free_memory
    Inspection   parameter_table, layer_summary, draw_head_graph
    Plots        plot_training_curves, plot_calibration, plot_test_results
    Hands-on     ask
"""
import gc
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
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
    print("   state:", row["state"][:state_chars].replace("\n", " "), "…")
    for i, choice in enumerate(row["choices"]):
        print(f"   {'→' if i == row['label'] else ' '} {i}. {choice[:110]}")
    print()


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
    questions = load_questions(split, max_questions, config["seed"])
    path = Path(ckpt_dir) / f"logits_{split}_{len(questions)}.pt"
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


def draw_head_graph(network, num_candidates=3):
    """torchview + Graphviz drawing of the decision head: every operation with its tensor shapes.
    Uses random candidate vectors for 2 questions (the second one with a padded, masked-out slot)."""
    from torchview import draw_graph
    H = network.backbone.config.hidden_size
    mask = torch.ones(2, num_candidates, dtype=torch.bool)
    mask[1, -1] = False
    head = network.head
    device = next(head.parameters()).device
    graph = draw_graph(head.cpu(), input_data={"choice_embeddings": torch.randn(2, num_candidates, H), "choice_mask": mask},
                       depth=3, expand_nested=True, graph_name=type(head).__name__)
    head.to(device)
    return graph.visual_graph


# ----------------------------------------------------------------------------------------- plots

def plot_training_curves(run_dir):
    """Train and validation loss / accuracy from metrics.jsonl; the dashed line is the best checkpoint."""
    history = read_metrics(run_dir)
    train, val = history["train"], history["validation"]
    best_step = json.loads((Path(run_dir) / "checkpoints" / "best" / "meta.json").read_text())["step"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, key in [(axes[0], "loss"), (axes[1], "accuracy")]:
        ax.plot([r["step"] for r in train], [r[key] for r in train], label="train", alpha=0.8)
        ax.plot([r["step"] for r in val], [r[key] for r in val], "o-", label="validation")
        ax.set_title(key)
    for task in TASK_TYPES:
        points = [(r["step"], r[f"accuracy_{task}"]) for r in val if f"accuracy_{task}" in r]
        axes[2].plot(*zip(*points), "o-", label=task)
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
    fig, axes = plt.subplots(1, len(before), figsize=(5.3 * len(before), 4.5))
    for ax, task in zip(axes, before):
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
