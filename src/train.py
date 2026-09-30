"""
Training loop for the from-scratch MiniGPT.

Resumable: each run continues from the last checkpoint, so a full training
run can be split across several shorter invocations without losing progress.

Run:
    python src/train.py --target_total 2000

Trains on data/input.txt (tiny-shakespeare by default — swap in your own
corpus for a more original portfolio piece, e.g. Indian news headlines,
Telugu/English code-switched text, or your own domain of interest).
"""

import os
import time
import argparse
import torch

from model import MiniGPT, count_parameters
from data import load_data, get_batch

# ---- Config -----------------------------------------------------------
BLOCK_SIZE = 128
BATCH_SIZE = 32
EMBED_DIM = 192
NUM_HEADS = 6
NUM_LAYERS = 4
DROPOUT = 0.1

MAX_ITERS = 2000
EVAL_INTERVAL = 200
EVAL_ITERS = 50
LEARNING_RATE = 3e-4

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "input.txt")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoint.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main(extra_iters=MAX_ITERS, target_total=None):
    print(f"Using device: {DEVICE}")

    train_data, val_data, tokenizer = load_data(DATA_PATH)
    print(f"Vocab size: {tokenizer.vocab_size} | Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
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
    stop_at = target_total if target_total is not None else start_step + extra_iters

    start = time.time()
    it = start_step
    while it < stop_at:
        if it % EVAL_INTERVAL == 0 or it == stop_at - 1:
            losses = estimate_loss(model, train_data, val_data)
            elapsed = time.time() - start
            print(f"step {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | {elapsed:.1f}s")

        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        it += 1

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": it,
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "block_size": BLOCK_SIZE,
                "embed_dim": EMBED_DIM,
                "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS,
            },
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
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
