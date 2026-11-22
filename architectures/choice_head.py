"""Decision heads: turn one embedding per choice into one logit per choice.

Both heads start with self-attention between the choices (no position embeddings, so the result
does not depend on the order of the choices). They differ in how a choice becomes a score:

bilinear_answer  (BEV, the original)  [task token, choice_1 ... choice_C, answer] -> self-attention
                 -> logit_i = <proj(answer), proj(choice_i)>. Needs an "answer" embedding.
candidate_mlp    (ours)               [choice_1 ... choice_C] -> self-attention
                 -> logit_i = MLP(choice_i), the same MLP for every choice. No answer embedding.

Pick one with head.type in the config. Both heads have the same call signature:
    head(choice_embeddings [B, C, H], choice_mask [B, C], task_type [B], answer_embedding [B, H] or None)
"""
import math

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, new_dim):
        super().__init__()
        self.new_dim = new_dim
        self.q = nn.Linear(hidden_dim, new_dim)
        self.k = nn.Linear(hidden_dim, new_dim)
        self.v = nn.Linear(hidden_dim, new_dim)
        self.o = nn.Linear(new_dim, hidden_dim)

    def forward(self, x, padding_mask):

        # x: [B, T, H], padding_mask: [B, T] (True = real token)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # compute attention weights, padded choices are never attended to
        scores = torch.einsum("bqd,bkd->bqk", [q, k])/math.sqrt(self.new_dim) # [B, T, T]
        scores = scores.masked_fill(~padding_mask[:, None, :], -float("inf"))
        weights = torch.softmax(scores, dim=-1)

        out = torch.einsum("bqk,bkd->bqd", [weights, v])

        return self.o(out)


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, new_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = SelfAttention(hidden_dim, new_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, new_dim),
            nn.GELU(),
            nn.Linear(new_dim, hidden_dim),
        )

    def forward(self, x, padding_mask):
        x = x + self.attn(self.norm1(x), padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class BilinearAnswerHead(nn.Module):
    """BEV head: self-attention over [task token, choice embeddings..., answer embedding], then a bilinear score."""

    def __init__(self, hidden_dim, new_dim, num_layers, num_task_types):
        super().__init__()
        self.new_dim = new_dim
        self.task_embedding = nn.Embedding(num_task_types, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([AttentionBlock(hidden_dim, new_dim) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.answer_proj = nn.Linear(hidden_dim, new_dim)
        self.choice_proj = nn.Linear(hidden_dim, new_dim)

    def forward(self, choice_embeddings, choice_mask, task_type, answer_embedding):
        # choice_embeddings: [B, C, H], answer_embedding: [B, H], choice_mask: [B, C], task_type: [B]
        B = choice_embeddings.shape[0]
        task_token = self.task_embedding(task_type).unsqueeze(1)

        x = torch.cat([task_token, choice_embeddings, answer_embedding.unsqueeze(1)], dim=1) # [B, C+2, H]
        x = self.input_norm(x)

        always_valid = torch.ones(B, 1, dtype=torch.bool, device=choice_mask.device)
        padding_mask = torch.cat([always_valid, choice_mask, always_valid], dim=1)
        for layer in self.layers:
            x = layer(x, padding_mask)
        x = self.final_norm(x)

        # Bilinear score between the answer slot and every choice slot -> works for any number of choices
        answer = self.answer_proj(x[:, -1])        # [B, D]
        choices = self.choice_proj(x[:, 1:-1])     # [B, C, D]
        logits = torch.einsum("bd,bcd->bc", [answer, choices])/math.sqrt(self.new_dim)
        return logits.masked_fill(~choice_mask, -float("inf"))


class CandidateMLPHead(nn.Module):
    """Our head: self-attention between the candidates only, then the same MLP scores each candidate.

        h_1 ... h_C  --self-attention x num_layers-->  h'_1 ... h'_C  --MLP (shared)-->  z_1 ... z_C

    num_layers=0 skips the interaction: each candidate is scored on its own (a useful ablation).
    task_type is not used: the task hint is already in the text every candidate was encoded with.
    """

    def __init__(self, hidden_dim, new_dim, num_layers, num_task_types=None):
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([AttentionBlock(hidden_dim, new_dim) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, new_dim),
            nn.GELU(),
            nn.Linear(new_dim, 1),
        )

    def forward(self, choice_embeddings, choice_mask, task_type=None, answer_embedding=None):
        x = self.input_norm(choice_embeddings)        # [B, C, H]
        for layer in self.layers:
            x = layer(x, choice_mask)                 # candidates attend to each other, never to padding
        x = self.final_norm(x)
        logits = self.scorer(x).squeeze(-1)           # [B, C]
        return logits.masked_fill(~choice_mask, -float("inf"))


HEADS = {"bilinear_answer": BilinearAnswerHead, "candidate_mlp": CandidateMLPHead}


def build_head(head_config, hidden_dim, num_task_types, default_type):
    head_config = dict(head_config)
    head_type = head_config.pop("type", default_type)
    if head_type not in HEADS:
        raise ValueError(f"head.type must be one of {list(HEADS)}, got {head_type!r}")
    return HEADS[head_type](hidden_dim=hidden_dim, num_task_types=num_task_types, **head_config)
