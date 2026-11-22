"""Post-training temperature scaling (Guo et al., 2017, "On Calibration of Modern Neural Networks").

A trained classifier is often over-confident: it says 95% when it is right 80% of the time.
Temperature scaling fixes this after training, without retraining, with one scalar per task type:

    p = softmax(logits / T_task)     T > 1 softens the probabilities, T < 1 sharpens them

Each T is chosen to minimise the negative log-likelihood on the *calibration* split, which is
held out from training AND from checkpoint selection (see dataset.py). Dividing a question's
logits by T never changes its argmax, so accuracy is unchanged; only the confidence moves.
One T per task type, because choice, yes/no (noul) and score questions have different confidence profiles.

Usage (writes calibration.json into the checkpoint; load_checkpoint and answer() then use it):
    uv run python calibration.py runs/<run-id>/checkpoints/best
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import TASK_TYPES

CALIBRATION_FILE = "calibration.json"


def fit_temperature(logits, labels, t_min=0.05, t_max=20.0, iters=60):
    """logits: [N, C] (padded choices = -inf), labels: [N]. Returns the NLL-optimal temperature in [t_min, t_max].

    NLL is convex in 1/T, so a golden-section search over log T finds the minimum reliably.
    Padded -inf logits are replaced by a large finite negative value: they still get probability 0,
    but -inf / T would turn into NaN during the search.
    """
    logits = logits.detach().float().cpu()
    logits = logits.masked_fill(torch.isinf(logits), -1e4)
    labels = labels.cpu()
    nll = lambda log_t: F.cross_entropy(logits / math.exp(log_t), labels).item()

    lo, hi = math.log(t_min), math.log(t_max)
    ratio = (math.sqrt(5) - 1) / 2
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = nll(a), nll(b)
    for _ in range(iters):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = nll(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = nll(b)
    return math.exp((lo + hi) / 2)


def fit_temperatures(logits, labels, task_types, min_questions=30):
    """One temperature per task type. A task with fewer than min_questions examples gets the global T."""
    global_t = fit_temperature(logits, labels)
    temperatures = {}
    for name, idx in TASK_TYPES.items():
        mask = task_types == idx
        temperatures[name] = fit_temperature(logits[mask], labels[mask]) if mask.sum() >= min_questions else global_t
    return temperatures


def save_temperatures(ckpt_dir, temperatures, **info):
    with open(Path(ckpt_dir) / CALIBRATION_FILE, "w") as f:
        json.dump({"temperatures": temperatures, **info}, f, indent=2)


def load_temperatures(ckpt_dir):
    """{task: T} from calibration.json, or 1.0 (no scaling) if the checkpoint has not been calibrated."""
    path = Path(ckpt_dir) / CALIBRATION_FILE
    if not path.exists():
        return 1.0
    saved = json.loads(path.read_text())
    return saved.get("temperatures", saved.get("temperature", 1.0))


def report(before, after):
    keys = [k for k in before if k.startswith(("loss", "nll", "ece", "brier", "mae", "accuracy", "auc"))]
    lines = [f"{'metric':<16}{'T = 1':>10}{'calibrated':>12}"]
    lines += [f"{k:<16}{before[k]:>10.4f}{after[k]:>12.4f}" for k in keys]
    return "\n".join(lines)


def main():
    from architectures import get_architecture
    from dataset import QuestionDataset, load_questions
    from inference import collect_logits, get_device
    from logger import load_checkpoint
    from metrics import compute_metrics

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="runs/<exp_id>/checkpoints/<name>")
    parser.add_argument("--split", default="calibration", help="never fit on the split you select checkpoints or report results on")
    parser.add_argument("--max_questions", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = get_device()
    network, config = load_checkpoint(args.checkpoint, device)
    arch, data = get_architecture(config), config["data"]
    dataset = QuestionDataset(load_questions(args.split, args.max_questions), arch.encode_example,
                              data["max_state_tokens"], data["max_choice_tokens"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=arch.collate_fn)
    logits, labels, task_types = collect_logits(network, loader, device)

    temperatures = fit_temperatures(logits, labels, task_types)
    before, after = compute_metrics(logits, labels, task_types), compute_metrics(logits, labels, task_types, temperatures)
    print("temperatures:", {k: round(v, 3) for k, v in temperatures.items()})
    print(report(before, after))
    save_temperatures(args.checkpoint, temperatures, split=args.split, num_questions=len(labels), before=before, after=after)
    print(f"Saved -> {Path(args.checkpoint) / CALIBRATION_FILE}")


if __name__ == "__main__":
    main()
