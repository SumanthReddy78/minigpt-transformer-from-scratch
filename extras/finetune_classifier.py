"""
Fine-tune a pretrained transformer (DistilBERT) for sentiment classification.

This is the "applied" half of the project: instead of training attention
from scratch, you take a model that already understands language and adapt
it to a specific task — the workflow you'll use constantly as an ML Engineer.

Needs internet access (to download the pretrained model + dataset from
HuggingFace), so run this on your own machine / Colab / Kaggle, not in a
network-restricted sandbox.

Run:
    python src/finetune_classifier.py

Requires: pip install transformers datasets scikit-learn accelerate
"""

import numpy as np
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./sentiment-model"

# Swap "imdb" for your own dataset (e.g. a resume-vs-job-description match
# dataset from Kaggle) to make this a more original portfolio piece.
DATASET_NAME = "imdb"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    cm = confusion_matrix(labels, preds)

    print("Confusion matrix:\n", cm)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    # Use a subset for a fast fine-tuning run; drop the slicing for the full run
    train_ds = dataset["train"].shuffle(seed=42).select(range(2000))
    eval_ds = dataset["test"].shuffle(seed=42).select(range(500))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    eval_ds = eval_ds.map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved fine-tuned model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
