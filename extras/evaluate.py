"""
Standalone evaluation of the fine-tuned classifier: Precision/Recall/F1,
confusion matrix, ROC-AUC — the same metric set already on your resume,
applied to the transformer project.

Run (after finetune_classifier.py has produced ./sentiment-model):
    python src/evaluate.py
"""

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "./sentiment-model"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    dataset = load_dataset("imdb")["test"].shuffle(seed=42).select(range(500))

    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for i in range(0, len(dataset), 32):
            batch = dataset[i : i + 32]
            enc = tokenizer(batch["text"], truncation=True, padding=True, max_length=256, return_tensors="pt")
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            all_labels.extend(batch["label"])

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()
