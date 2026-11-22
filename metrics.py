"""Evaluation metrics, reported per task type because the three types use the probabilities differently.

all tasks   loss (NLL) and accuracy, overall and per task (nll_<task>, accuracy_<task>)
choice      ece_choice: calibration of the top-choice confidence
noul        the quantity we use is P(yes): auc_noul (ranking), brier_noul and ece_noul (calibration of P(yes))
score       the quantity we use is the expected score sum_i i * p_i: mae_score, plus ece_score (top-level confidence)

ECE uses 10 bins and is noisy on small sets, so read it next to NLL / Brier.
"""
import torch
import torch.nn.functional as F

from dataset import TASK_NAMES, TASK_TYPES


def apply_temperature(logits, task_types, temperature=1.0):
    """temperature: one float for every question, or a dict {task name: T}."""
    if isinstance(temperature, dict):
        t = torch.tensor([temperature.get(TASK_NAMES[int(k)], 1.0) for k in task_types], dtype=logits.dtype)
        return logits / t[:, None]
    return logits / temperature


def binary_auc(scores, labels):
    """Probability that a random positive gets a higher score than a random negative (ties count half)."""
    scores, labels = torch.as_tensor(scores), torch.as_tensor(labels, dtype=torch.bool)
    positives, negatives = scores[labels], scores[~labels].sort().values
    below = torch.searchsorted(negatives, positives, right=False)
    below_or_tied = torch.searchsorted(negatives, positives, right=True)
    return (below + below_or_tied).sum().item() / 2 / (len(positives) * len(negatives))


def reliability_bins(probability, outcome, n_bins=10):
    """Groups predictions by predicted probability. Per bin: mean predicted probability, observed frequency, count."""
    probability, outcome = torch.as_tensor(probability).float(), torch.as_tensor(outcome).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    bins = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (probability >= lo) & (probability <= hi) if i == 0 else (probability > lo) & (probability <= hi)
        if in_bin.any():
            bins.append({"lo": lo.item(), "hi": hi.item(), "predicted": probability[in_bin].mean().item(),
                         "observed": outcome[in_bin].mean().item(), "count": int(in_bin.sum())})
    return bins


def expected_calibration_error(probability, outcome, n_bins=10):
    """Average |observed - predicted| over probability bins, weighted by bin size. 0 = perfectly calibrated."""
    n = len(probability)
    return sum(b["count"] / n * abs(b["observed"] - b["predicted"]) for b in reliability_bins(probability, outcome, n_bins))


def compute_metrics(logits, labels, task_types, temperature=1.0):
    """logits [N, C] (padded choices = -inf), labels [N], task_types [N]; temperature: float or {task: T}."""
    logits = apply_temperature(logits, task_types, temperature)
    nll = F.cross_entropy(logits, labels, reduction="none")
    probs = torch.softmax(logits, dim=-1)
    confidence, prediction = probs.max(-1)
    hits = prediction == labels

    metrics = {"loss": nll.mean().item(), "accuracy": hits.float().mean().item()}
    for name, idx in TASK_TYPES.items():
        mask = task_types == idx
        if not mask.any():
            continue
        metrics[f"accuracy_{name}"] = hits[mask].float().mean().item()
        metrics[f"nll_{name}"] = nll[mask].mean().item()
        if name == "noul":
            p_yes, is_yes = probs[mask, 1], labels[mask].float()
            metrics["brier_noul"] = ((p_yes - is_yes) ** 2).mean().item()
            metrics["ece_noul"] = expected_calibration_error(p_yes, is_yes)
            if 0 < is_yes.sum() < len(is_yes):
                metrics["auc_noul"] = binary_auc(p_yes, is_yes.bool())
        else:
            metrics[f"ece_{name}"] = expected_calibration_error(confidence[mask], hits[mask])
        if name == "score":
            expected = (probs[mask] * torch.arange(probs.shape[1])).sum(-1)
            metrics["mae_score"] = (expected - labels[mask]).abs().mean().item()
    return metrics
