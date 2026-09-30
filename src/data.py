"""
Character-level tokenizer + batching for the from-scratch transformer.

Character-level keeps the vocabulary tiny (~65 tokens) so the model in model.py
trains fast on CPU. Swapping in a BPE tokenizer (e.g. tiktoken) is a natural
"v2" upgrade to mention in an interview if asked how you'd scale this up.
"""

import torch


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_data(path: str, val_fraction: float = 0.1):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = int(len(data) * (1 - val_fraction))
    train_data, val_data = data[:n], data[n:]
    return train_data, val_data, tokenizer


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str = "cpu"):
    """Sample a random batch of (input, target) sequences for next-token prediction."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
