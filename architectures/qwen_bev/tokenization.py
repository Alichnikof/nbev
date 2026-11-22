"""BEV encoding: the question and all choices go into ONE Qwen sequence.

Choices share starting position ids and the attention mask hides them from each other,
so each choice is read as if it were the only one after the question.
"""
import os

import torch
from transformers import AutoTokenizer

from architectures.prompts import context_prefix

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # the tokenizer is shared with DataLoader worker processes

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_prompt(question,choices):
    choice_begin_token = tokenizer.encode("<option>")
    choice_end_token = tokenizer.encode("</option>")
    answer_prompt = "The answer is:"

    question_tokens = tokenizer.encode(question)
    answer_tokens = tokenizer.encode(answer_prompt)

    choice_tokens = [
        tokenizer.encode(choice) for choice in choices
    ]

    final_tokens = []
    final_tokens.extend(question_tokens)

    span = {
        "question_end_idx": len(question_tokens) - 1,
        "choices_end_idx": [],
        "answer_end_idx": 0
    }

    for choice in choice_tokens:
        final_tokens.extend(choice_begin_token)
        final_tokens.extend(choice)
        final_tokens.extend(choice_end_token)
        span["choices_end_idx"].append(len(final_tokens) - 1)

    final_tokens.extend(answer_tokens)
    span["answer_end_idx"] = len(final_tokens) - 1

    return final_tokens, span


def generate_position_ids(span):
    position_ids = []

    start_idx = 0
    for i in range(span["question_end_idx"] + 1):
        position_ids.append(start_idx)
        start_idx += 1

    next_choice_begin_idx = start_idx

    for choice_end_idx in span["choices_end_idx"]:
        for idx in range(choice_end_idx - next_choice_begin_idx + 1):
            position_ids.append(start_idx + idx)

        next_choice_begin_idx = choice_end_idx + 1

    # Answer tokens start right after the longest choice, so every choice token is in their past
    answer_start_idx = max(position_ids) + 1
    for idx in range(span["answer_end_idx"] + 1 - len(position_ids)):
        position_ids.append(answer_start_idx + idx)

    return position_ids


def get_attention_mask(span):
    L = span["answer_end_idx"] + 1
    attention_mask = torch.zeros((L, L))

    question_len = span["question_end_idx"] + 1

    # Question tokens look back at everything in the past (within the question)
    attention_mask[:question_len, :question_len] = torch.tril(torch.ones(question_len, question_len))

    # Data structure that stores the spans of the choices (begin, end of each choices)
    choice_spans = []
    choice_start_idx = question_len
    for choice_end_idx in span["choices_end_idx"]:
        choice_spans.append((choice_start_idx, choice_end_idx))
        choice_start_idx = choice_end_idx + 1


    # Looped over each span of the choices and set the attention mask for the span
    for choice_start_idx, choice_end_idx in choice_spans:


        # Each choice tokens looks back within that choice span (using causal mask)
        attention_mask[
            choice_start_idx:choice_end_idx + 1,
            choice_start_idx:choice_end_idx + 1
        ] = torch.tril(torch.ones(choice_end_idx - choice_start_idx + 1,
                                choice_end_idx - choice_start_idx + 1)
                        )

    # Choice tokens look at question tokens
    attention_mask[question_len:, :question_len] = 1

    # Answer tokens looks back at everything in the past
    answer_tokens_begin = choice_spans[-1][1] + 1
    attention_mask[answer_tokens_begin:, :answer_tokens_begin] = 1
    attention_mask[answer_tokens_begin:, answer_tokens_begin:] = torch.tril(torch.ones(L - answer_tokens_begin, L - answer_tokens_begin))
    return attention_mask


def truncate(text, max_tokens):
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens])


def build_question(state, instructions, task_type, max_state_tokens):
    return f"{context_prefix(instructions, task_type)}{truncate(state, max_state_tokens)}\n"


def encode_example(state, instructions, choices, task_type, max_state_tokens, max_choice_tokens):
    question = build_question(state, instructions, task_type, max_state_tokens)
    choices = [truncate(choice, max_choice_tokens) for choice in choices]

    final_tokens, span = tokenize_prompt(question, choices)
    position_ids = generate_position_ids(span)
    attention_mask = get_attention_mask(span)

    # Choice embeddings are read at the last content token of each choice, not at the closing ">" of </option>
    choice_end_token_len = len(tokenizer.encode("</option>"))
    choice_read_idx = [end_idx - choice_end_token_len for end_idx in span["choices_end_idx"]]

    return {
        "input_ids": final_tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "choice_read_idx": choice_read_idx,
        "answer_end_idx": span["answer_end_idx"],
    }


def collate_fn(batch):
    B = len(batch)
    L = max(len(ex["input_ids"]) for ex in batch)
    C = max(len(ex["choice_read_idx"]) for ex in batch)

    input_ids = torch.full((B, L), tokenizer.pad_token_id, dtype=torch.long)
    position_ids = torch.zeros((B, L), dtype=torch.long)
    # Padded query rows attend only to themselves, otherwise their softmax row is all -inf -> NaN
    attention_mask = torch.eye(L).repeat(B, 1, 1)
    choice_read_idx = torch.zeros((B, C), dtype=torch.long)
    choice_mask = torch.zeros((B, C), dtype=torch.bool)
    answer_end_idx = torch.zeros(B, dtype=torch.long)

    for b, ex in enumerate(batch):
        n, c = len(ex["input_ids"]), len(ex["choice_read_idx"])
        input_ids[b, :n] = torch.tensor(ex["input_ids"])
        position_ids[b, :n] = torch.tensor(ex["position_ids"])
        attention_mask[b, :n, :n] = ex["attention_mask"]
        choice_read_idx[b, :c] = torch.tensor(ex["choice_read_idx"])
        choice_mask[b, :c] = True
        answer_end_idx[b] = ex["answer_end_idx"]

    attention_mask = torch.where(attention_mask == 0, -float("inf"), 0.0).unsqueeze(1)  # B, 1, L, L

    out = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "choice_read_idx": choice_read_idx,
        "choice_mask": choice_mask,
        "answer_end_idx": answer_end_idx,
        "task_type": torch.tensor([ex["task_type"] for ex in batch], dtype=torch.long),
    }
    if all(ex["label"] is not None for ex in batch):
        out["labels"] = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
    return out
