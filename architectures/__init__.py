"""Two ways to turn (state, question, choices) into one logit per choice. Both end in a decision head.

Encoders (model.architecture):

qwen_bev       Qwen decoder (causal). ONE sequence per question:
                   question+state <option>A</option> <option>B</option> ... The answer is:
               Choices share position ids and are masked from each other, so the context is encoded once.
               Choice embedding = hidden state of the last token of each choice.

deberta_cross  DeBERTa encoder (bidirectional). ONE sequence per (question, choice) pair:
                   [CLS] question+state [SEP] choice [SEP]
               The context is re-encoded for every choice (cross-encoder).
               Choice embedding = [CLS] or the mean of the choice tokens (config: model.pooling).

nbev           "not-bev": DeBERTa encoder (bidirectional). ONE sequence per question, BEV-style:
                   [CLS] question+state [SEP] [OPT] A [SEP] [OPT] B [SEP] ...
               Choices share positions and are masked from each other, so the context is encoded once
               and the result does not depend on choice order.
               Choice embedding = the [OPT] token, the mean of the choice tokens, or both (config: model.pooling).

Decision heads (head.type, see choice_head.py), usable with any encoder:
    bilinear_answer  BEV: choices attend to each other, then are scored against an "answer" embedding
    candidate_mlp    ours: choices attend to each other, then the same MLP scores each one
Defaults: qwen_bev -> bilinear_answer, deberta_cross and nbev -> candidate_mlp.

Each architecture package exposes the same three functions:
    encode_example(state, instructions, choices, task_type, max_state_tokens, max_choice_tokens) -> dict
    collate_fn(examples) -> batch dict of tensors, always incl. choice_mask [B, C], task_type [B], labels [B]
    build_network(config, num_task_types) -> nn.Module with a .backbone and a .head;
                                             forward(**batch) returns logits [B, C] (padded choices = -inf)
"""
import importlib

ARCHITECTURES = ("qwen_bev", "deberta_cross", "nbev")
DEFAULT_ARCHITECTURE = "qwen_bev"  # configs without model.architecture are the original BEV setup


def architecture_name(config):
    return config["model"].get("architecture") or DEFAULT_ARCHITECTURE


def get_architecture(config):
    name = architecture_name(config)
    if name not in ARCHITECTURES:
        raise ValueError(f"unknown model.architecture {name!r}, expected one of {ARCHITECTURES}")
    # Imported lazily so only the selected model's tokenizer is downloaded and loaded
    return importlib.import_module(f"architectures.{name}")
