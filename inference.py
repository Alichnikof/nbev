import argparse

import torch
import torch.nn.functional as F

from architectures import get_architecture
from dataset import SPLITS, TASK_TYPES, QuestionDataset, load_questions, question_to_choices
from logger import load_checkpoint
from metrics import compute_metrics


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device):
    # bf16 keeps fp32's exponent range, so no loss scaling is needed (unlike fp16)
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type in ("cuda", "mps"))


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def collect_logits(network, dataloader, device):
    """Runs the network over a loader. Returns logits [N, C_max] (padded choices = -inf), labels [N], task_types [N]."""
    network.eval()
    all_logits, labels, task_types = [], [], []
    for batch in dataloader:
        batch = to_device(batch, device)
        with autocast(device):
            all_logits.append(network(**batch).float().cpu())
        labels.append(batch["labels"].cpu())
        task_types.append(batch["task_type"].cpu())
    network.train()
    C = max(l.shape[1] for l in all_logits)
    logits = torch.cat([F.pad(l, (0, C - l.shape[1]), value=-float("inf")) for l in all_logits])
    return logits, torch.cat(labels), torch.cat(task_types)


def evaluate(network, dataloader, device, temperature=1.0):
    return compute_metrics(*collect_logits(network, dataloader, device), temperature)


@torch.no_grad()
def answer(network, config, state, question, device):
    """Answers one question dict (dataset / TypeSafe format) with a typed answer. config = the checkpoint's config.

    Probabilities use network.temperature: {task: T} from calibration.json via load_checkpoint, or 1.0 if not calibrated."""
    network.eval()
    arch, data = get_architecture(config), config["data"]
    choices, _ = question_to_choices(question)
    example = arch.encode_example(state, question["instructions"], choices, question["type"],
                                  data["max_state_tokens"], data["max_choice_tokens"])
    example.update(task_type=TASK_TYPES[question["type"]], label=None)
    batch = to_device(arch.collate_fn([example]), device)

    with autocast(device):
        logits = network(**batch)[0].float()
    temperature = getattr(network, "temperature", 1.0)
    if isinstance(temperature, dict):
        temperature = temperature[question["type"]]
    probs = torch.softmax(logits / temperature, dim=-1).tolist()

    if question["type"] == "choice":
        keys = list(question["criteria"])
        return {"choice": keys[max(range(len(keys)), key=probs.__getitem__)], "probabilities": dict(zip(keys, probs))}
    if question["type"] == "score":
        return {"score": sum(i * p for i, p in enumerate(probs)), "probabilities": probs}
    return {"noul": probs[1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="runs/<exp_id>/checkpoints/<name>")
    parser.add_argument("--split", default="test", choices=SPLITS)
    parser.add_argument("--max_questions", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    network, config = load_checkpoint(args.checkpoint, device)
    arch, data = get_architecture(config), config["data"]
    questions = load_questions(args.split, args.max_questions, task_types=data.get("task_types"))
    # Same encoding and truncation as during training, read from the checkpoint's config
    dataset = QuestionDataset(questions, arch.encode_example, data["max_state_tokens"], data["max_choice_tokens"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=arch.collate_fn,
                                         num_workers=args.num_workers)
    print(evaluate(network, loader, device, temperature=network.temperature))


if __name__ == "__main__":
    main()
