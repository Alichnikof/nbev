import hashlib
import json

import torch
from datasets import load_dataset

DATASET_NAME = "avbiswas/bev-decision-150K"
TASK_TYPES = {"choice": 0, "noul": 1, "score": 2}
TASK_NAMES = {v: k for k, v in TASK_TYPES.items()}
# ~4% of the dataset's train split is held out and cut in two halves (~5K questions each):
#   validation   picks the best checkpoint during training
#   calibration  fits the temperatures after training
# The split is a hash of the state TEXT: some states appear in several rows, and hashing sends every
# copy to the same split, so no state leaks between train, validation and calibration. Deterministic.
HELDOUT_PER_MILLE = {"validation": range(0, 20), "calibration": range(20, 40)}  # buckets out of 1000
SPLITS = ("train", "validation", "calibration", "test")


def state_bucket(state):
    return int(hashlib.md5(state.encode()).hexdigest(), 16) % 1000


def question_to_choices(question):
    """Turns any of the 3 question types into a list of option strings (+ label index if present)."""
    if question["type"] == "choice":
        keys = list(question["criteria"])
        choices = [f"{key}: {question['criteria'][key]}" if question["criteria"][key] else key for key in keys]
        label = keys.index(question["label"]) if "label" in question else None
    elif question["type"] == "score":
        # Choices share position ids, so the level index is written into the text to keep the ordering
        choices = [f"{i}: {level}" for i, level in enumerate(question["criteria"])]
        label = question.get("label")
    else:  # noul -> P(Yes)
        choices = ["No", "Yes"]
        label = int(question["label"]) if "label" in question else None
    return choices, label


def explode_questions(rows):
    """One dataset row can hold many questions about the same state -> one output row per question."""
    out = {"state": [], "task_type": [], "instructions": [], "choices": [], "label": []}
    for state, questions_json in zip(rows["state"], rows["questions_json"]):
        for question in json.loads(questions_json).values():
            choices, label = question_to_choices(question)
            out["state"].append(state)
            out["task_type"].append(TASK_TYPES[question["type"]])
            out["instructions"].append(question["instructions"])
            out["choices"].append(choices)
            out["label"].append(label)
    return out


def load_questions(split, max_questions=None, seed=0):
    """split: one of SPLITS. validation and calibration are held out of the dataset's train split (see above)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if split == "test":
        ds = load_dataset(DATASET_NAME, split="test")
    else:
        heldout = range(0, 1000) if split == "train" else HELDOUT_PER_MILLE[split]
        keep = (lambda b: b >= 40) if split == "train" else (lambda b: b in heldout)
        ds = load_dataset(DATASET_NAME, split="train").filter(lambda row: keep(state_bucket(row["state"])))
    ds = ds.shuffle(seed=seed)
    if max_questions is not None:
        # Every row has at least one question, so this many rows is always enough
        ds = ds.select(range(min(max_questions, len(ds))))
    ds = ds.map(explode_questions, batched=True, remove_columns=ds.column_names)
    if max_questions is not None:
        ds = ds.select(range(min(max_questions, len(ds))))
    return ds


class QuestionDataset(torch.utils.data.Dataset):
    """Architecture-agnostic: encode_fn is the selected architecture's encode_example."""

    def __init__(self, questions, encode_fn, max_state_tokens, max_choice_tokens):
        self.questions = questions
        self.encode_fn = encode_fn
        self.max_state_tokens = max_state_tokens
        self.max_choice_tokens = max_choice_tokens

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        row = self.questions[idx]
        example = self.encode_fn(
            row["state"], row["instructions"], row["choices"], TASK_NAMES[row["task_type"]],
            self.max_state_tokens, self.max_choice_tokens,
        )
        example["task_type"] = row["task_type"]
        example["label"] = row["label"]
        return example
