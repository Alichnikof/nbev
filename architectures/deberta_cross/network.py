"""Cross-encoder network: DeBERTa embeds each (context, choice) sequence, a decision head compares them.

Pooling (config model.pooling) decides which vector represents a choice:
    cls          hidden state of [CLS]: a summary of the whole (context, choice) pair
    choice_mean  mean of the choice's own token states, already contextualised by the question and state

Default head: candidate_mlp (self-attention between candidates, then the same MLP scores each one).
The BEV head (bilinear_answer) also works here, but it needs an "answer" embedding and an encoder has
no "The answer is:" token, so this network then adds `answer_query`: one learned vector of size H,
used as the answer embedding for every question.
"""
import torch
import torch.nn as nn
from transformers import AutoModel

from architectures.choice_head import CandidateMLPHead, build_head
from architectures.finetuning import setup_finetuning

POOLINGS = ("cls", "choice_mean")


class DebertaCrossNetwork(nn.Module):
    def __init__(self, backbone, head, pooling):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"model.pooling must be one of {POOLINGS}, got {pooling!r}")
        self.backbone = backbone
        self.head = head
        self.pooling = pooling
        # Only the bilinear (BEV) head needs an answer embedding
        self.answer_query = None if isinstance(head, CandidateMLPHead) else nn.Parameter(torch.randn(backbone.config.hidden_size) * 0.02)

    def forward(self, input_ids, attention_mask, choice_token_mask, seq_question_idx, seq_choice_idx,
                choice_mask, task_type, **kwargs):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()  # [N, L, H]

        if self.pooling == "cls":
            seq_embeddings = hidden[:, 0]                                                  # [N, H]
        else:
            weights = choice_token_mask.unsqueeze(-1).float()
            seq_embeddings = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1)       # [N, H]

        # Regroup the N flat sequences into [B, C, H]; padded choice slots stay zero and are masked in the head
        B, C = choice_mask.shape
        choice_embeddings = hidden.new_zeros(B, C, hidden.shape[-1])
        choice_embeddings = choice_embeddings.index_put((seq_question_idx, seq_choice_idx), seq_embeddings)
        answer_embedding = None if self.answer_query is None else self.answer_query.expand(B, -1)

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
    backbone = load_backbone(model["name"], model.get("num_layers"))
    target = lora_target_regex(backbone.config.num_hidden_layers, lora["last_k_layers"])
    backbone = setup_finetuning(backbone, lora, target, model.get("full_finetune", False))
    head = build_head(config["head"], backbone.config.hidden_size, num_task_types, default_type="candidate_mlp")
    return DebertaCrossNetwork(backbone, head, model.get("pooling", "choice_mean"))
