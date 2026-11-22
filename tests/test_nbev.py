"""Checks for nbev and the state truncation. Runs on CPU in about a minute (loads deberta-v3-base).

    uv run python -m tests.test_nbev
"""
import torch
from transformers import AutoModel

from architectures.nbev import build_network, collate_fn, encode_example
from architectures.nbev.tokenization import tokenizer

STATE = "User: my laptop will not turn on after the update.\nResponse: Hold the power button for 10 seconds."
QUESTION = "Which team should handle this request?"
CHOICES = ["Technical support: hardware and software problems", "Billing", "Other"]
CONFIG = {
    "model": {"name": "microsoft/deberta-v3-base", "num_layers": 2, "pooling": "opt_mean", "full_finetune": True},
    "lora": {"last_k_layers": 0, "r": 8, "alpha": 16, "dropout": 0.0},
    "head": {"type": "candidate_mlp", "new_dim": 64, "num_layers": 2},
}


def batch_of(*questions):
    examples = []
    for state, choices in questions:
        example = encode_example(state, QUESTION, choices, "choice", max_state_tokens=448, max_choice_tokens=64)
        examples.append({**example, "task_type": 0, "label": 0})
    return collate_fn(examples)


def logits_of(network, *questions):
    with torch.no_grad():
        return network(**batch_of(*questions))


def test_order_invariance(network):
    # Shuffling the choices must shuffle the logits the same way, and nothing else
    order = [2, 0, 1]
    original = logits_of(network, (STATE, CHOICES))[0]
    shuffled = logits_of(network, (STATE, [CHOICES[i] for i in order]))[0]
    assert torch.allclose(original[order], shuffled, atol=1e-4), (original, shuffled)


def test_choices_never_see_each_other(network):
    # Changing choice B must leave choice A's token states unchanged (the head may still compare them later)
    def states_of_first_choice(choices):
        batch = batch_of((STATE, choices))
        with torch.no_grad():
            hidden = network.encode(batch["input_ids"], batch["position_ids"], batch["padding_mask"], batch["attention_mask"])
        start = batch["opt_idx"][0, 0]
        return hidden[0, start:start + 5]
    before = states_of_first_choice(CHOICES)
    after = states_of_first_choice([CHOICES[0], "A completely different second option, much longer than before", "Other"])
    assert torch.allclose(before, after, atol=1e-5)


def test_batching_and_padding(network):
    # A question gets the same logits alone or in a batch with a longer question (padding is invisible)
    long_state = STATE + " The fan makes a clicking noise." * 30
    alone = logits_of(network, (STATE, CHOICES))[0]
    batched = logits_of(network, (STATE, CHOICES), (long_state, CHOICES[:2]))[0]
    assert torch.allclose(alone, batched, atol=1e-4), (alone, batched)


def test_patched_attention_matches_stock_deberta(network):
    # With ordinary positions 0..L-1 and full attention, the per-question position code must reproduce stock DeBERTa
    stock = AutoModel.from_pretrained(CONFIG["model"]["name"], num_hidden_layers=2, dtype=torch.float32).eval()
    a = tokenizer.encode("The quick brown fox jumps over the lazy dog, twice in a row.")
    b = tokenizer.encode("Another sentence of a similar length, give or take a word or two.")
    L = min(len(a), len(b))
    ids = torch.tensor([a[:L], b[:L]])  # two different questions -> two different position-bias computations
    padding = torch.ones_like(ids)
    full = torch.ones(2, L, L, dtype=torch.bool)
    positions = torch.arange(L).expand(2, L)
    with torch.no_grad():
        ours = network.encode(ids, positions, padding, full)
        reference = stock(input_ids=ids, attention_mask=padding).last_hidden_state
    network_weights = network.deberta.state_dict()
    stock_weights = stock.state_dict()
    assert all(torch.equal(network_weights[k], stock_weights[k]) for k in stock_weights)  # same pretrained weights
    assert torch.allclose(ours, reference, atol=1e-5), (ours - reference).abs().max()


def test_long_state_keeps_its_end():
    from architectures.deberta_cross.tokenization import encode_example as cross_encode
    from architectures.qwen_bev.tokenization import build_question
    long_state = "Prompt: hello\n" + "User: tell me more.\nAssistant: sure, here is more.\n" * 200 + "Response: THE-END-MARKER"
    end_ids = tokenizer.encode("THE-END-MARKER", add_special_tokens=False)

    def contains(ids, sub):
        return any(ids[i:i + len(sub)] == sub for i in range(len(ids) - len(sub) + 1))
    packed = encode_example(long_state, QUESTION, CHOICES, "choice", 448, 64)
    cross = cross_encode(long_state, QUESTION, CHOICES, "choice", 448, 64)
    assert contains(packed["input_ids"], end_ids) and contains(cross["sequences"][0], end_ids)
    assert packed["context_len"] <= 448 + 40  # state budget + [CLS], [SEP] and the short question prefix
    question = build_question(long_state, QUESTION, "choice", 448)
    assert "THE-END-MARKER" in question and "Prompt: hello" in question


if __name__ == "__main__":
    torch.manual_seed(0)
    network = build_network(CONFIG, num_task_types=3).eval()
    tests = [test_order_invariance, test_choices_never_see_each_other, test_batching_and_padding,
             test_patched_attention_matches_stock_deberta]
    for test in tests:
        test(network)
        print(f"ok  {test.__name__}")
    test_long_state_keeps_its_end()
    print("ok  test_long_state_keeps_its_end")
