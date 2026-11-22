"""qwen_bev: Qwen decoder, one sequence per question, choices masked from each other (the original BEV setup)."""
from .network import build_network
from .tokenization import collate_fn, encode_example
