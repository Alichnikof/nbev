"""BEV network: a (truncated) Qwen decoder reads the joint sequence, a decision head scores the choices.

Default head: bilinear_answer, scoring each choice against the hidden state of "The answer is:".
"""
import torch
import torch.nn as nn
from transformers import Qwen3Model

from architectures.choice_head import build_head
from architectures.finetuning import setup_finetuning


class QwenBEVNetwork(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids, position_ids, attention_mask, choice_read_idx, choice_mask, answer_end_idx, task_type, **kwargs):
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=False
        )
        hidden = output.last_hidden_state.float()
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)

        choice_embeddings = hidden[batch_idx[:, None], choice_read_idx]   # [B, C, H]
        answer_embedding = hidden[batch_idx, answer_end_idx]              # [B, H]
        # Head runs in fp32 even under bf16 autocast: bf16 logits are too coarse for a stable loss
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self.head(choice_embeddings, choice_mask, task_type, answer_embedding)


def load_backbone(model_name, num_layers=None):
    # Only the first num_layers decoder layers are built and loaded; the rest are never executed
    overrides = {} if num_layers is None else {"num_hidden_layers": num_layers}
    return Qwen3Model.from_pretrained(model_name, dtype=torch.float32, **overrides)


def lora_target_regex(num_layers, last_k_layers):
    layer_ids = "|".join(str(i) for i in range(num_layers - last_k_layers, num_layers))
    return rf".*layers\.({layer_ids})\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"


def build_network(config, num_task_types):
    model, lora = config["model"], config["lora"]
    backbone = load_backbone(model["name"], model.get("num_layers"))
    target = lora_target_regex(backbone.config.num_hidden_layers, lora["last_k_layers"])
    backbone = setup_finetuning(backbone, lora, target, model.get("full_finetune", False))
    head = build_head(config["head"], backbone.config.hidden_size, num_task_types, default_type="bilinear_answer")
    return QwenBEVNetwork(backbone, head)
