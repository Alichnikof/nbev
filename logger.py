import json
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

from architectures import architecture_name, get_architecture
from calibration import load_temperatures
from dataset import TASK_TYPES


class RunLogger:
    """Each training run gets runs/<name>-<timestamp>/ with config, metrics and checkpoints (trainable weights only)."""

    def __init__(self, config, run_name, device, root="runs"):
        self.config = config
        self.exp_id = f"{run_name}-{datetime.now():%Y%m%d-%H%M%S}"
        self.run_dir = Path(root) / self.exp_id
        self.run_dir.mkdir(parents=True)
        self.start_time = time.time()

        self.write_config(self.run_dir)
        with open(self.run_dir / "run.json", "w") as f:
            json.dump({"exp_id": self.exp_id, "architecture": architecture_name(config), "device": str(device),
                       "started_at": datetime.now().isoformat()}, f, indent=2)
        print(f"Experiment {self.exp_id} -> {self.run_dir}")

    def write_config(self, directory):
        with open(directory / "config.yaml", "w") as f:
            yaml.safe_dump(self.config, f, sort_keys=False)

    def log(self, step, **metrics):
        record = {"step": step, "time": round(time.time() - self.start_time, 1), **metrics}
        with open(self.run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()))

    def save_checkpoint(self, network, name="final", step=None):
        """A checkpoint folder is self-contained: copy it anywhere and load_checkpoint() rebuilds the network.

        Only trainable tensors are saved (LoRA adapters or fully fine-tuned weights, head, answer query);
        frozen pretrained weights are re-downloaded from the Hugging Face Hub when loading.
        """
        ckpt_dir = self.run_dir / "checkpoints" / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        weights = {k: p.detach().cpu().contiguous() for k, p in network.named_parameters() if p.requires_grad}
        save_file(weights, ckpt_dir / "weights.safetensors")

        self.write_config(ckpt_dir)
        with open(ckpt_dir / "meta.json", "w") as f:
            json.dump({"exp_id": self.exp_id, "step": step, "architecture": architecture_name(self.config)}, f, indent=2)
        print(f"Saved checkpoint -> {ckpt_dir}")
        return ckpt_dir

    def finish(self, **summary):
        summary["duration_sec"] = round(time.time() - self.start_time, 1)
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


def load_trainable_weights(network, ckpt_dir):
    """Load a checkpoint into a network built from the same config (same architecture, LoRA and head)."""
    weights = load_file(Path(ckpt_dir) / "weights.safetensors")
    result = network.load_state_dict(weights, strict=False)
    trainable = {k for k, p in network.named_parameters() if p.requires_grad}
    missing = sorted(trainable - weights.keys())
    if result.unexpected_keys or missing:
        raise ValueError(f"{ckpt_dir} doesn't match this config: unexpected={result.unexpected_keys[:3]} missing={missing[:3]}")


def load_checkpoint(ckpt_dir, device):
    with open(Path(ckpt_dir) / "config.yaml") as f:
        config = yaml.safe_load(f)
    network = get_architecture(config).build_network(config, len(TASK_TYPES))
    load_trainable_weights(network, ckpt_dir)
    network.temperature = load_temperatures(ckpt_dir)  # {task: T}, or 1.0 until calibration.py has run
    return network.to(device).eval(), config
