"""nbev ("new BEV"): DeBERTa encoder, ONE sequence per question; choices share positions and are masked apart (as in BEV)."""
from .network import build_network
from .tokenization import collate_fn, encode_example
