"""deberta_cross: DeBERTa encoder, one sequence per (question, choice) pair, choices compared in the head."""
from .network import build_network
from .tokenization import collate_fn, encode_example
