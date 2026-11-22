# Training JEV networks from Qwen

Train a network that answers choice, yes/no (`noul`), and score questions using a Qwen backbone and a small attention head. Choices share the same starting position IDs, and the attention mask prevents each choice from seeing the others. The head scores the choices without depending on their order.

[Watch the video: Training JEV networks from Qwen](https://youtu.be/sF3CNPbWA8o)

If you find this helpful, consider supporting on Patreon — it hosts all code, projects, slides, and write-ups from the YouTube channel.

[<img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patron!" width="200">](https://www.patreon.com/NeuralBreakdownwithAVB)

## Start here: `arch.ipynb`

Open [`arch.ipynb`](arch.ipynb) for the architecture walkthrough from the livestream: tokenize the question and choices separately, build their position IDs, and construct the attention mask. The notebook explains the architecture; the Python files implement training and inference.

- [`architectures/`](architectures/): the two networks being compared (see below).
- [`dataset.py`](dataset.py): load the dataset and expand rows into typed questions. Splits: `train`, `validation` (checkpoint selection), `calibration` (temperature fitting) and `test`. They are split by state text, so no state appears in two splits.
- [`train.py`](train.py): training, gradient accumulation, validation, and checkpoint initialization.
- [`logger.py`](logger.py): run metrics and checkpoints (trainable weights only).
- [`inference.py`](inference.py): evaluation and typed answers through `answer()`.
- [`metrics.py`](metrics.py): per-task metrics: accuracy, NLL, P(yes) Brier score and AUC, expected-score MAE, ECE.
- [`calibration.py`](calibration.py): post-training temperature scaling, one temperature per task type.
- [`deberta_full_walkthrough.ipynb`](deberta_full_walkthrough.ipynb): our DeBERTa model end to end, from data to training, calibration, test and hands-on questions. It uses [`configs/deberta_walkthrough.yaml`](configs/deberta_walkthrough.yaml) and reuses a finished run instead of retraining.
- [`notebook_utils.py`](notebook_utils.py): the notebook's helpers (train or reuse a run, cached evaluation, plots, `ask`).

## Architectures

Pick an encoder with `model.architecture` and a decision head with `head.type`. Everything else is shared: data, prompt text, training loop, and metrics.

| | `qwen_bev` (original) | `deberta_cross` | `nbev` |
|---|---|---|---|
| Backbone | Qwen3 decoder (causal) | DeBERTa-v3 encoder (bidirectional) | DeBERTa-v3 encoder (bidirectional) |
| Sequences per question | 1: `question+state <option>A</option><option>B</option>… The answer is:` | 1 per choice: `[CLS] question+state [SEP] choice [SEP]` | 1: `[CLS] question+state [SEP] [OPT] A [SEP] [OPT] B [SEP] …` |
| Choice order | invariant: choices share positions and are masked apart | invariant: one sequence per choice | invariant: choices share positions and are masked apart, as in BEV |
| Choice embedding | last token of each choice | `model.pooling`: `cls` or `choice_mean` | `model.pooling`: `opt` (the `[OPT]` token), `mean` or `opt_mean` |
| Default head | `bilinear_answer` | `candidate_mlp` | `candidate_mlp` |
| Cost | context encoded once | context encoded once per choice | context encoded once (~3x faster than `deberta_cross`) |
| Code | [`architectures/qwen_bev/`](architectures/qwen_bev/) | [`architectures/deberta_cross/`](architectures/deberta_cross/) | [`architectures/nbev/`](architectures/nbev/) |

`nbev` ("new BEV") is BEV's packing on a bidirectional encoder. Every choice restarts at the same position, which on its own makes a bidirectional encoder order-invariant. The attention mask lets a choice see only the context and itself, so each choice is read as if it were the only one. Without the mask, tokens of different choices would share positions and the encoder could not tell which choice a token belongs to: "not good" + "very bad" would be the same input as "very good" + "not bad". Its `[OPT]` token (DeBERTa's `[MASK]` token, reused) attends to all of the choice's tokens, so its output represents the whole choice however many tokens it has. DeBERTa-v3 only uses relative positions, so the shared positions are passed as a per-question relative-position matrix; [`network.py`](architectures/nbev/network.py) patches the attention layers to accept one matrix per question. `uv run python -m tests.test_nbev` checks order invariance, choice isolation, padding, and that the patch reproduces stock DeBERTa.

Long states keep their first 64 tokens and their end; the middle is dropped (all architectures). States end with the `Response:` being judged, so cutting the end would hide it from the model.

Decision heads ([`choice_head.py`](architectures/choice_head.py)) work with any encoder. In both, the candidates first attend to each other with no positional encoding, so the result doesn't depend on candidate order:

- `bilinear_answer` (BEV): self-attention over `[task token, candidates…, answer]`, then each candidate is scored by a bilinear product with the answer slot. With `deberta_cross`, which has no answer token, the answer slot is a learned vector (`answer_query`).
- `candidate_mlp` (ours): self-attention over the candidates only, then the same MLP turns each candidate into a logit. `head.num_layers: 0` scores each candidate independently.

Each package has a `tokenization.py` (`encode_example`, `collate_fn`) and a `network.py` (`build_network`). Backbone training is set per config: `lora.last_k_layers: 0` freezes it, `k > 0` adds LoRA to the last k layers, and `model.full_finetune: true` trains every weight.

### Comparing them

[`configs/compare/`](configs/compare/) holds Mac-sized runs on identical data: 10K training questions, 448 state tokens, 16 questions per optimizer step. Qwen trains with LoRA as in BEV; DeBERTa is fully fine-tuned, as a cross-encoder (`cls` or `choice_mean` pooling) or packed (`nbev.yaml`). They batch questions of similar length together (`data.group_by_length`) and keep the checkpoint with the lowest validation NLL (`logging.select_by: loss`; the default is `accuracy`). Each takes up to about an hour on an M5 Pro:

```bash
uv run python train.py configs/compare/deberta_mean.yaml --name cmp-deberta-mean
uv run python calibration.py runs/cmp-deberta-mean-<timestamp>/checkpoints/best
```

Full-scale DeBERTa configs for a GPU: [`deberta_lora.yaml`](configs/deberta_lora.yaml) and [`deberta_full_ft.yaml`](configs/deberta_full_ft.yaml). Sanity check: [`deberta_smoke.yaml`](configs/deberta_smoke.yaml).

Dataset: [avbiswas/bev-decision-150K](https://huggingface.co/datasets/avbiswas/bev-decision-150K). Training loads this dataset automatically.

Original livestream: [Architecture walkthrough](https://youtube.com/live/AzxoU7kxjig).

## Setup

Install [uv](https://docs.astral.sh/uv/), then install the locked dependencies:

```bash
uv sync
```

The first run downloads the Qwen3-0.6B weights, tokenizer, and dataset. The code selects CUDA, then Apple MPS, then CPU. Full training configs are intended for a GPU; reduce the batch size for smaller devices.

For a fresh Ubuntu GPU machine, [`cloud_startup_script.sh`](cloud_startup_script.sh) installs the system tools, sets up uv, and checks CUDA availability:

```bash
bash cloud_startup_script.sh
```

## Train

For a small overfitting sanity check, use the 10-question config:

```bash
uv run python train.py configs/smoke.yaml --name smoke
```

This config runs 30 epochs to check whether the model can learn those questions. For full training with the first 20 Qwen layers and LoRA on the last 12 kept layers (8–19):

```bash
uv run python train.py configs/full_lora_k12.yaml --name full-lora-k12
```

That config trains the choice head and LoRA adapters, uses 1,024 state tokens, and accumulates two microbatches of 32 questions per optimizer step. `optim.grad_accum_steps` defaults to 1. Logging, validation, and saving intervals count optimizer steps. An incomplete final accumulation group is skipped.

Other configs include [`full_head_only.yaml`](configs/full_head_only.yaml) for a frozen backbone and [`full_lora.yaml`](configs/full_lora.yaml) for LoRA on the last four of 20 kept layers.

Each run writes its config, metrics, summary, and checkpoints under `runs/<name>-<timestamp>/`. A checkpoint holds `weights.safetensors` (every trainable tensor: head, LoRA or fine-tuned weights) and its `config.yaml`; frozen pretrained weights are loaded from the Hugging Face Hub. `best` tracks test accuracy, `latest` is overwritten at each save interval, and `final` is saved at the end.

## Start from saved weights

Pass a checkpoint directory or a run ID (which selects that run's final checkpoint):

```bash
uv run python train.py configs/full_lora_k12.yaml --name continued \
  --resume runs/<run-id>/checkpoints/final
```

Use the same architecture, backbone, head, and LoRA configuration as the checkpoint. `--resume` initializes a new training run from the saved weights: the optimizer, learning-rate schedule, and step counter start fresh.

## Evaluate

```bash
uv run python inference.py runs/<run-id>/checkpoints/final \
  --max_questions 100 --batch_size 2 --num_workers 0
```

Evaluation reports loss, overall accuracy, accuracy per task type, yes/no AUC when both label classes are present, and expected calibration error (ECE). It uses the checkpoint's architecture and token limits. `--split` is `train`, `validation`, or `test`.

## Calibrate

The model's probabilities are used directly: P(yes) for yes/no questions and the expected score for score questions. These numbers are only meaningful if the model is calibrated. Accuracy and AUC can't show calibration problems, because they only use the ranking of the probabilities.

After training, fit one temperature per task type on the `calibration` split, which is used neither for training nor for checkpoint selection. Then `softmax(logits / T_task)` gives calibrated probabilities, and accuracy doesn't change. The temperatures are saved as `calibration.json` in the checkpoint, and `load_checkpoint`, `inference.py` and `answer()` use them automatically from then on:

```bash
uv run python calibration.py runs/<run-id>/checkpoints/best --max_questions 2000
```
