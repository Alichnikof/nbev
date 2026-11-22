"""Cross-encoder encoding: ONE DeBERTa sequence per (question, choice) pair.

    [CLS] task hint + question + state [SEP] choice tokens [SEP]

Every choice sees the full context with bidirectional attention but never sees the other
choices; comparing choices is left to the ChoiceHead. A question with C choices becomes
C sequences, so the context is encoded C times (the price of this design).
"""
import os

import torch
from transformers import AutoTokenizer

from architectures.prompts import context_prefix

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # the tokenizer is shared with DataLoader worker processes

# deberta-v3-xsmall/small/base/large all share this SentencePiece vocabulary
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def encode_example(state, instructions, choices, task_type, max_state_tokens, max_choice_tokens):
    context_ids = (tokenizer.encode(context_prefix(instructions, task_type), add_special_tokens=False)
                   + tokenizer.encode(state, add_special_tokens=False)[:max_state_tokens])

    sequences, choice_spans = [], []
    for choice in choices:
        choice_ids = tokenizer.encode(choice, add_special_tokens=False)[:max_choice_tokens]
        sequences.append([tokenizer.cls_token_id, *context_ids, tokenizer.sep_token_id, *choice_ids, tokenizer.sep_token_id])
        choice_start = len(context_ids) + 2  # after [CLS] context [SEP]
        choice_spans.append((choice_start, choice_start + len(choice_ids)))  # [start, end) of the choice tokens

    return {"sequences": sequences, "choice_spans": choice_spans}


def collate_fn(batch):
    """B questions with C_b choices each -> N = sum(C_b) flat sequences, plus the indices to regroup them."""
    B = len(batch)
    C = max(len(ex["sequences"]) for ex in batch)
    flat = [(b, c, ids, span) for b, ex in enumerate(batch)
            for c, (ids, span) in enumerate(zip(ex["sequences"], ex["choice_spans"]))]
    N, L = len(flat), max(len(ids) for _, _, ids, _ in flat)

    input_ids = torch.full((N, L), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((N, L), dtype=torch.long)      # 1 = real token, padding is ignored
    choice_token_mask = torch.zeros((N, L), dtype=torch.bool)   # True on the choice's own tokens (for mean pooling)
    seq_question_idx = torch.zeros(N, dtype=torch.long)         # which question each sequence belongs to
    seq_choice_idx = torch.zeros(N, dtype=torch.long)           # and which of its choices it is
    choice_mask = torch.zeros((B, C), dtype=torch.bool)

    for n, (b, c, ids, (start, end)) in enumerate(flat):
        input_ids[n, :len(ids)] = torch.tensor(ids)
        attention_mask[n, :len(ids)] = 1
        choice_token_mask[n, start:end] = True
        seq_question_idx[n], seq_choice_idx[n] = b, c
        choice_mask[b, c] = True

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "choice_token_mask": choice_token_mask,
        "seq_question_idx": seq_question_idx,
        "seq_choice_idx": seq_choice_idx,
        "choice_mask": choice_mask,
        "task_type": torch.tensor([ex["task_type"] for ex in batch], dtype=torch.long),
    }
    if all(ex["label"] is not None for ex in batch):
        out["labels"] = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
    return out
