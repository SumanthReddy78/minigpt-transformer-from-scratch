"""
Phase B: fine-tune the pretrained backbone (from pretrain_imdb_lm.py) for
sentiment classification on real, labeled IMDB reviews.

Run (after pretrain_imdb_lm.py has produced imdb_backbone.pt):
    python src/finetune_imdb_classifier.py
"""

import os
import time
import argparse
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)

from classifier_model import TransformerClassifier
from imdb_data import load_imdb, get_classification_batch, iterate_eval_batches

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "imdb_reviews.csv")
BACKBONE_PATH = os.path.join(os.path.dirname(__file__), "..", "imdb_backbone.pt")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "imdb_classifier.pt")

MAX_LEN = 250
VOCAB_SIZE = 4000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EVAL_INTERVAL = 100
GRAD_CLIP = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate(model, ids, lengths, labels, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    for b_ids, b_mask, b_labels in iterate_eval_batches(ids, lengths, labels, batch_size=64, device=device):
        logits = model(b_ids, b_mask)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(b_labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    model.train()
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "roc_auc": auc, "confusion_matrix": cm}


def main(extra_iters, target_total=None, n_train=20000, n_test=5000, lr=LEARNING_RATE):
    print(f"Using device: {DEVICE}")

    # vocab_fit_size defaults to 6000, matching the vocabulary the backbone's
    # token embeddings were pretrained against — n_train can grow independently
    # without desynchronizing token ids from those embeddings.
    data = load_imdb(DATA_PATH, max_len=MAX_LEN, vocab_size=VOCAB_SIZE,
                      n_train=n_train, n_test=n_test, build_lm_stream=False)
    train_ids, train_lengths, train_labels = data["train"]
    test_ids, test_lengths, test_labels = data["test"]
    print(f"Train: {train_ids.shape[0]} reviews | Test: {test_ids.shape[0]} reviews")

    model, stoi, itos, cfg = TransformerClassifier.from_pretrained_checkpoint(
        BACKBONE_PATH, device=DEVICE, target_block_size=MAX_LEN
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    start_step = 0
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        for g in optimizer.param_groups:
            g["lr"] = lr  # allow overriding LR on resume, e.g. for a decayed final phase
        start_step = ckpt["step"]
        print(f"Resumed fine-tuning from step {start_step}")

    stop_at = target_total if target_total is not None else start_step + extra_iters

    start = time.time()
    it = start_step
    while it < stop_at:
        b_ids, b_mask, b_labels = get_classification_batch(train_ids, train_lengths, train_labels, BATCH_SIZE, DEVICE)
        logits = model(b_ids, b_mask)
        loss = criterion(logits, b_labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if it % EVAL_INTERVAL == 0 or it == stop_at - 1:
            elapsed = time.time() - start
            print(f"step {it:4d} | train loss {loss.item():.4f} | {elapsed:.1f}s")
        if it > 0 and it % 500 == 0:
            quick_metrics = evaluate(model, test_ids, test_lengths, test_labels, DEVICE)
            print(f"         -> test accuracy so far: {quick_metrics['accuracy']:.4f} (F1 {quick_metrics['f1']:.4f})")
        it += 1

    torch.save(
        {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "step": it,
         "vocab_stoi": stoi, "vocab_itos": itos, "backbone_config": cfg},
        CHECKPOINT_PATH,
    )
    print(f"Saved classifier checkpoint at step {it}")

    print("\nEvaluating on held-out test set...")
    metrics = evaluate(model, test_ids, test_lengths, test_labels, DEVICE)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iters", type=int, default=1200)
    parser.add_argument("--target_total", type=int, default=None)
    parser.add_argument("--n_train", type=int, default=20000)
    parser.add_argument("--n_test", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    main(extra_iters=args.max_iters, target_total=args.target_total, n_train=args.n_train, n_test=args.n_test, lr=args.lr)
