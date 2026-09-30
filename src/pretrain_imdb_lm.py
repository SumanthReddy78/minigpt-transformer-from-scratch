"""
Phase A of the fine-tuning pipeline: pretrain the from-scratch transformer
via next-token prediction on real (unlabeled) IMDB review text.

This is "domain-adaptive pretraining" — the same idea behind further
pretraining a language model on in-domain text before fine-tuning it on a
downstream task. Here it also doubles as proof the backbone is reusable
across two different objectives (language modeling, then classification).

Run:
    python src/pretrain_imdb_lm.py
"""

import os
import time
import argparse
import torch

from model import MiniGPT, count_parameters
from data import get_batch  # generic sliding-window batcher, reused as-is
from imdb_data import load_imdb

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "imdb_reviews.csv")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "imdb_backbone.pt")

# Config sized for CPU-only training in a few minutes
MAX_LEN = 100
VOCAB_SIZE = 4000
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 3
BATCH_SIZE = 32
MAX_ITERS = 2000
EVAL_INTERVAL = 100
LEARNING_RATE = 3e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(extra_iters=MAX_ITERS, target_total=None):
    """Trains for `extra_iters` more steps beyond whatever is already in the
    checkpoint (resumable — each call is capped at a few minutes of wall
    clock, so a full run is done across several calls without losing
    progress). Pass target_total to stop once that many cumulative steps
    have been reached, regardless of extra_iters."""
    print(f"Using device: {DEVICE}")
    data = load_imdb(DATA_PATH, max_len=MAX_LEN, vocab_size=VOCAB_SIZE,
                      n_train=6000, n_test=1500, n_extra_unlabeled=6000)
    vocab = data["vocab"]
    token_stream = data["lm_token_stream"]
    n = int(len(token_stream) * 0.95)
    train_stream, val_stream = token_stream[:n], token_stream[n:]

    print(f"Vocab size: {len(vocab)} | LM corpus tokens: {len(token_stream):,}")

    model = MiniGPT(
        vocab_size=len(vocab),
        block_size=MAX_LEN,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    start_step = 0
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from checkpoint at step {start_step}")

    print(f"Model parameters: {count_parameters(model):,}")

    if target_total is not None:
        stop_at = target_total
    else:
        stop_at = start_step + extra_iters

    start = time.time()
    it = start_step
    while it < stop_at:
        x, y = get_batch(train_stream, MAX_LEN, BATCH_SIZE, DEVICE)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if it % EVAL_INTERVAL == 0 or it == stop_at - 1:
            with torch.no_grad():
                vx, vy = get_batch(val_stream, MAX_LEN, BATCH_SIZE, DEVICE)
                _, val_loss = model(vx, vy)
            elapsed = time.time() - start
            print(f"step {it:4d} | train loss {loss.item():.4f} | val loss {val_loss.item():.4f} | {elapsed:.1f}s")
        it += 1

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": it,
            "config": {
                "vocab_size": len(vocab),
                "block_size": MAX_LEN,
                "embed_dim": EMBED_DIM,
                "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS,
            },
            "vocab_itos": vocab.itos,
            "vocab_stoi": vocab.stoi,
        },
        CHECKPOINT_PATH,
    )
    print(f"Saved checkpoint at step {it} to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iters", type=int, default=MAX_ITERS,
                         help="additional steps to run beyond the checkpoint")
    parser.add_argument("--target_total", type=int, default=None,
                         help="stop once this many cumulative steps are reached")
    args = parser.parse_args()
    main(extra_iters=args.max_iters, target_total=args.target_total)
