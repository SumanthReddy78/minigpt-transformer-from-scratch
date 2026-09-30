"""
Real-data pipeline for the sentiment classification half of the project.

Uses the standard 50,000-review IMDB sentiment dataset (25k positive /
25k negative), word-level tokenized with a frequency-capped vocabulary —
kept small on purpose so the from-scratch transformer trains fast on CPU.

Sequences are LEFT-padded: pad tokens come first, real tokens are
right-aligned. Combined with causal attention, this means the final
position in every sequence is always a real token whose representation
has attended to the entire review — exactly the pooling trick used in
the original GPT paper (Radford et al., 2018) for fine-tuning a
decoder-only transformer on classification tasks.
"""

import re
import random
from collections import Counter

import pandas as pd
import torch

PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"


def clean_text(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text)          # strip HTML line breaks
    text = re.sub(r"[^a-zA-Z0-9' ]", " ", text)      # drop punctuation/symbols
    return text.lower().strip()


def tokenize(text: str) -> list[str]:
    return clean_text(text).split()


class Vocab:
    def __init__(self, texts: list[str], max_size: int = 8000, min_freq: int = 2):
        counter = Counter()
        for t in texts:
            counter.update(tokenize(t))

        most_common = [w for w, c in counter.most_common(max_size) if c >= min_freq]
        self.itos = [PAD_TOKEN, UNK_TOKEN] + most_common
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str, max_len: int) -> tuple[list[int], int]:
        ids = [self.stoi.get(w, self.unk_id) for w in tokenize(text)]
        true_len = min(len(ids), max_len)
        ids = ids[:max_len]
        pad_needed = max_len - len(ids)
        ids = [self.pad_id] * pad_needed + ids  # LEFT padding
        return ids, true_len


def load_imdb(csv_path: str, max_len: int = 200, vocab_size: int = 8000,
              vocab_fit_size: int = 6000,
              n_train: int = 6000, n_test: int = 1500, n_extra_unlabeled: int = 6000,
              seed: int = 42, build_lm_stream: bool = True):
    """
    vocab_fit_size is fixed independently of n_train so that re-running this
    with a larger n_train (more labeled fine-tuning data) still builds the
    IDENTICAL vocabulary — same word -> id mapping — that an already-trained
    backbone's token embeddings were pretrained against. Growing n_train
    without this would silently misalign token ids with the wrong embeddings.
    """
    df = pd.read_csv(csv_path)
    df["label"] = (df["sentiment"] == "positive").astype(int)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    vocab_fit_df = df.iloc[:vocab_fit_size]
    vocab = Vocab(vocab_fit_df["review"].tolist(), max_size=vocab_size)

    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train : n_train + n_test]
    extra_df = df.iloc[n_train + n_test : n_train + n_test + n_extra_unlabeled]

    def encode_split(split_df):
        encoded = [vocab.encode(r, max_len) for r in split_df["review"]]
        ids = torch.tensor([e[0] for e in encoded], dtype=torch.long)
        lengths = torch.tensor([e[1] for e in encoded], dtype=torch.long)
        labels = torch.tensor(split_df["label"].tolist(), dtype=torch.long)
        return ids, lengths, labels

    train_ids, train_lengths, train_labels = encode_split(train_df)
    test_ids, test_lengths, test_labels = encode_split(test_df)

    lm_token_stream = None
    if build_lm_stream:
        # Flatten train + extra-unlabeled text into one long token stream for
        # LM pretraining (same sliding-window sampling as the char-level model)
        lm_texts = train_df["review"].tolist() + extra_df["review"].tolist()
        lm_ids = []
        for t in lm_texts:
            lm_ids.extend(vocab.stoi.get(w, vocab.unk_id) for w in tokenize(t))
        lm_token_stream = torch.tensor(lm_ids, dtype=torch.long)

    return {
        "vocab": vocab,
        "train": (train_ids, train_lengths, train_labels),
        "test": (test_ids, test_lengths, test_labels),
        "lm_token_stream": lm_token_stream,
    }


def get_classification_batch(ids, lengths, labels, batch_size, device="cpu"):
    idx = torch.randint(0, ids.size(0), (batch_size,))
    batch_ids = ids[idx].to(device)
    batch_lengths = lengths[idx].to(device)
    batch_labels = labels[idx].to(device)

    # Attention mask: 1 for real tokens (right-aligned), 0 for left-padding
    seq_len = batch_ids.size(1)
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    pad_amount = (seq_len - batch_lengths).unsqueeze(1)
    attention_mask = (positions >= pad_amount).float()

    return batch_ids, attention_mask, batch_labels


def iterate_eval_batches(ids, lengths, labels, batch_size, device="cpu"):
    """Deterministic full pass over a split, for evaluation."""
    n = ids.size(0)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_ids = ids[start:end].to(device)
        batch_lengths = lengths[start:end].to(device)
        batch_labels = labels[start:end].to(device)

        seq_len = batch_ids.size(1)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pad_amount = (seq_len - batch_lengths).unsqueeze(1)
        attention_mask = (positions >= pad_amount).float()

        yield batch_ids, attention_mask, batch_labels
