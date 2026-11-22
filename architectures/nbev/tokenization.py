"""nbev encoding: ONE DeBERTa sequence per question, holding the context and every choice.

    [CLS] task hint + question + state [SEP]  [OPT] choice 1 [SEP]  [OPT] choice 2 [SEP]  ...
    positions: 0 ...................... n-1    n    n+1 ...           n    n+1 ...

Same idea as BEV (qwen_bev), adapted to a bidirectional encoder:
  - every choice restarts at the same position n, so no choice is "first" or "last". In a bidirectional
    encoder this alone makes the output order-invariant: shuffling the choices shuffles the logits the same way.
  - the attention mask keeps the choices apart:
        context tokens  -> context only (the context is encoded once, independently of the choices)
        choice block i  -> context + its own block ([OPT], its tokens, its [SEP])
    so each choice is read as if it were the only one after the question. Without it, tokens of different
    choices sit at the same positions and attention cannot tell which choice a token belongs to:
    "not good" + "very bad" becomes the same input as "very good" + "not bad", and every [OPT] token
    gets the same vector. (A causal model like Qwen also needs the mask for order invariance itself.)

[OPT] marks the start of each choice; its output vector represents the whole choice (like GLiNER's [ENT]
token), however many tokens the choice has. [OPT] reuses DeBERTa's [MASK] token, which never occurs
in the text, so the vocabulary stays unchanged.
"""
import os

import torch

from architectures.deberta_cross.tokenization import encode_context, tokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # the tokenizer is shared with DataLoader worker processes

OPT_TOKEN_ID = tokenizer.mask_token_id


def encode_example(state, instructions, choices, task_type, max_state_tokens, max_choice_tokens):
    context_ids = [tokenizer.cls_token_id, *encode_context(state, instructions, task_type, max_state_tokens),
                   tokenizer.sep_token_id]
    n = len(context_ids)

    input_ids, position_ids = list(context_ids), list(range(n))
    blocks = []  # (start, end) of each choice block [OPT] tokens [SEP], end exclusive
    for choice in choices:
        choice_ids = tokenizer.encode(choice, add_special_tokens=False)[:max_choice_tokens]
        block = [OPT_TOKEN_ID, *choice_ids, tokenizer.sep_token_id]
        blocks.append((len(input_ids), len(input_ids) + len(block)))
        input_ids += block
        position_ids += range(n, n + len(block))  # every choice restarts at position n

    return {"input_ids": input_ids, "position_ids": position_ids, "context_len": n, "blocks": blocks}


def attention_mask_for(example, L):
    """[L, L] bool, True = the row token may attend to the column token. Padding rows see only themselves."""
    n = example["context_len"]
    mask = torch.eye(L, dtype=torch.bool)
    mask[:n, :n] = True                       # context <-> context
    for start, end in example["blocks"]:
        mask[start:end, :n] = True            # choice -> context
        mask[start:end, start:end] = True     # choice <-> itself, never another choice
    return mask


def collate_fn(batch):
    B = len(batch)
    L = max(len(ex["input_ids"]) for ex in batch)
    C = max(len(ex["blocks"]) for ex in batch)

    input_ids = torch.full((B, L), tokenizer.pad_token_id, dtype=torch.long)
    position_ids = torch.zeros((B, L), dtype=torch.long)
    padding_mask = torch.zeros((B, L), dtype=torch.long)        # 1 = real token
    attention_mask = torch.zeros((B, L, L), dtype=torch.bool)
    opt_idx = torch.zeros((B, C), dtype=torch.long)              # where each choice's [OPT] token is
    choice_token_mask = torch.zeros((B, C, L), dtype=torch.bool)  # each choice's own tokens (for mean pooling)
    choice_mask = torch.zeros((B, C), dtype=torch.bool)

    for b, ex in enumerate(batch):
        n_tokens = len(ex["input_ids"])
        input_ids[b, :n_tokens] = torch.tensor(ex["input_ids"])
        position_ids[b, :n_tokens] = torch.tensor(ex["position_ids"])
        padding_mask[b, :n_tokens] = 1
        attention_mask[b] = attention_mask_for(ex, L)
        for c, (start, end) in enumerate(ex["blocks"]):
            opt_idx[b, c] = start
            choice_token_mask[b, c, start + 1:end - 1] = True  # between [OPT] and [SEP]
            choice_mask[b, c] = True

    out = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "padding_mask": padding_mask,
        "attention_mask": attention_mask,
        "opt_idx": opt_idx,
        "choice_token_mask": choice_token_mask,
        "choice_mask": choice_mask,
        "task_type": torch.tensor([ex["task_type"] for ex in batch], dtype=torch.long),
    }
    if all(ex["label"] is not None for ex in batch):
        out["labels"] = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
    return out
