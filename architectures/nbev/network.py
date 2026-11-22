"""nbev network: DeBERTa reads the whole packed sequence once, a decision head scores the choices.

DeBERTa-v3 has no absolute position embeddings: attention only sees the DISTANCE between two tokens
(relative positions). Shared positions are therefore given as a per-question relative-position matrix
    rel[b, q, k] = position[b, q] - position[b, k]
which the stock Hugging Face code does not support for batch size > 1; per_example_relative_positions()
patches each attention layer to compute its position bias one question at a time with the library's own code.

Pooling (config model.pooling) decides which vector represents a choice:
    opt          output of the choice's [OPT] token (default)
    mean         mean of the choice's own token states
    opt_mean     both, concatenated and projected back to the hidden size
"""
import types

import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.models.deberta_v2.modeling_deberta_v2 import DisentangledSelfAttention, make_log_bucket_position

from architectures.choice_head import CandidateMLPHead, build_head
from architectures.finetuning import setup_finetuning

POOLINGS = ("opt", "mean", "opt_mean")


def _per_example_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
    """disentangled_attention_bias that accepts relative_pos [B, L, L] (one matrix per question)."""
    if relative_pos is None or relative_pos.dim() != 3 or relative_pos.size(0) == 1:
        return DisentangledSelfAttention.disentangled_attention_bias(
            self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)
    H = self.num_attention_heads  # query_layer/key_layer are [B * H, L, d], question b owns rows b*H ... b*H+H-1
    return torch.cat([
        DisentangledSelfAttention.disentangled_attention_bias(
            self, query_layer[b * H:(b + 1) * H], key_layer[b * H:(b + 1) * H], relative_pos[b],
            rel_embeddings, scale_factor)
        for b in range(relative_pos.size(0))
    ])


def per_example_relative_positions(backbone):
    for module in backbone.modules():
        if isinstance(module, DisentangledSelfAttention):
            module.disentangled_attention_bias = types.MethodType(_per_example_bias, module)
    return backbone


class NBEVNetwork(nn.Module):
    def __init__(self, backbone, head, pooling):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"model.pooling must be one of {POOLINGS}, got {pooling!r}")
        self.backbone = backbone
        self.head = head
        self.pooling = pooling
        H = self.deberta.config.hidden_size
        self.pool_proj = nn.Linear(2 * H, H) if pooling == "opt_mean" else None
        # Only the bilinear (BEV) head needs an answer embedding (see deberta_cross/network.py)
        self.answer_query = None if isinstance(head, CandidateMLPHead) else nn.Parameter(torch.randn(H) * 0.02)

    @property
    def deberta(self):
        """The DebertaV2Model, also when the backbone is wrapped by PEFT (LoRA)."""
        return self.backbone.get_base_model() if hasattr(self.backbone, "get_base_model") else self.backbone

    def relative_positions(self, position_ids):
        encoder = self.deberta.encoder
        rel = position_ids[:, :, None] - position_ids[:, None, :]  # [B, L, L]
        if encoder.position_buckets > 0:  # DeBERTa-v3 buckets far distances logarithmically, as in its pretraining
            rel = make_log_bucket_position(rel, encoder.position_buckets, encoder.max_relative_positions)
        return rel.long()

    def encode(self, input_ids, position_ids, padding_mask, attention_mask):
        """Token states [B, L, H] of the packed sequences (stock DebertaV2Model.forward takes no position matrix)."""
        deberta = self.deberta
        embeddings = deberta.embeddings(input_ids=input_ids, mask=padding_mask)
        return deberta.encoder(embeddings, attention_mask, output_hidden_states=False,
                               relative_pos=self.relative_positions(position_ids)).last_hidden_state

    def forward(self, input_ids, position_ids, padding_mask, attention_mask, opt_idx, choice_token_mask,
                choice_mask, task_type, **kwargs):
        hidden = self.encode(input_ids, position_ids, padding_mask, attention_mask).float()  # [B, L, H]

        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        opt = hidden[batch_idx[:, None], opt_idx]                                  # [B, C, H]
        weights = choice_token_mask.float()                                        # [B, C, L]
        mean = torch.bmm(weights, hidden) / weights.sum(-1, keepdim=True).clamp(min=1)  # [B, C, H]
        if self.pooling == "opt":
            choice_embeddings = opt
        elif self.pooling == "mean":
            choice_embeddings = mean
        else:
            choice_embeddings = self.pool_proj(torch.cat([opt, mean], dim=-1))
        choice_embeddings = choice_embeddings * choice_mask.unsqueeze(-1)          # padded choice slots stay zero
        answer_embedding = None if self.answer_query is None else self.answer_query.expand(hidden.shape[0], -1)

        # Head runs in fp32 even under bf16 autocast: bf16 logits are too coarse for a stable loss
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self.head(choice_embeddings, choice_mask, task_type, answer_embedding)


def load_backbone(model_name, num_layers=None):
    overrides = {} if num_layers is None else {"num_hidden_layers": num_layers}
    return AutoModel.from_pretrained(model_name, dtype=torch.float32, **overrides)


def lora_target_regex(num_layers, last_k_layers):
    layer_ids = "|".join(str(i) for i in range(num_layers - last_k_layers, num_layers))
    return rf".*encoder\.layer\.({layer_ids})\.attention\.(self\.(query_proj|key_proj|value_proj)|output\.dense)"


def build_network(config, num_task_types):
    model, lora = config["model"], config["lora"]
    backbone = per_example_relative_positions(load_backbone(model["name"], model.get("num_layers")))
    target = lora_target_regex(backbone.config.num_hidden_layers, lora["last_k_layers"])
    backbone = setup_finetuning(backbone, lora, target, model.get("full_finetune", False))
    head = build_head(config["head"], backbone.config.hidden_size, num_task_types, default_type="candidate_mlp")
    return NBEVNetwork(backbone, head, model.get("pooling", "opt"))
