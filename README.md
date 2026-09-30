# Transformer From Scratch + Fine-Tuned Sentiment Classifier

Train it yourself, on CPU. No checkpoints are included — you run the training.

## Quick start

```bash
pip install -r requirements.txt
python run_all.py --skip_part1        # trains the classifier (~35-45 min on 1 CPU core)
```

Or step by step:

```bash
python src/pretrain_imdb_lm.py --target_total 2000          # Phase A: pretrain on IMDB text
python src/finetune_imdb_classifier.py --target_total 1200  # Phase B: fine-tune, prints accuracy
python src/train.py --target_total 2000                     # Part 1: Shakespeare generator (optional)
streamlit run app.py                                        # demo
```

Everything saves checkpoints and **resumes**: press Ctrl+C any time and re-run the
same command to continue. No GPU needed; PyTorch uses all your CPU cores automatically.

## What to expect

Timings measured on a single CPU core (a multi-core laptop should be equal or faster):

| Stage | Steps | Time | What you should see |
|---|---|---|---|
| Phase A: pretraining | 2,000 | ~15 min | loss 8.3 → ~4.9 |
| Phase B: fine-tuning | 1,200 | ~20 min | test accuracy **~82–85%** |
| Part 1: Shakespeare model (optional, separate from the classifier) | 2,000 | ~26 min | loss 4.1 → ~1.7 (val ~1.75) |

Fine-tuning accuracy on 5,000 held-out reviews from my own runs of this exact
pipeline (accuracy is printed every 500 steps and at the end):

| Fine-tune steps | Test accuracy |
|---|---|
| 500 | 82.3% |
| 800 | 84.0% |
| 1,100 | 84.4% |
| 1,400 | 84.7% |

That was a single end-to-end run, not an average over seeds. Training has
randomness, so yours will differ a little (expect low-to-mid 80s).
**If you land under 80%**, keep training — it resumes from where it stopped:

```bash
python src/finetune_imdb_classifier.py --target_total 2000
```

Accuracy plateaus around 84–85% at this model size, so don't expect much beyond that.

## How it works

- `src/model.py` — decoder-only transformer written from scratch: manual multi-head
  causal self-attention, positional embeddings, pre-norm residual blocks.
- **Phase A** (`pretrain_imdb_lm.py`): next-word prediction on ~2.8M tokens of real IMDB
  reviews (4,000-word vocabulary).
- **Phase B** (`finetune_imdb_classifier.py`): adds a classification head over a
  mean-pooled representation and fine-tunes on 20,000 labeled reviews; tests on 5,000
  held-out reviews. The context window is extended from 100 to 250 words by resizing
  the position-embedding table (the only weights tied to sequence length).
- Data: the 50,000-review IMDB dataset (`data/imdb_reviews.csv`), and public-domain
  Shakespeare (`data/input.txt`) for Part 1.
- The eval numbers are printed during training so you can watch them, but no
  checkpoint is chosen by them. For a stricter setup, carve out a separate validation split.

## A bug worth knowing about (good interview story)

Left-padded batches give padding positions zero valid attention keys, so `softmax`
over an all-`-inf` row yields NaN — and `0 × NaN = NaN`, so it poisons every later
layer and every gradient. Fix (in `model.py`): always let each position attend to itself.

## Resume bullets (fill in YOUR numbers after training)

- Implemented a decoder-only transformer from scratch in PyTorch (multi-head causal
  self-attention, positional embeddings, pre-norm residual blocks).
- Pretrained it on 2.8M tokens of IMDB text, then fine-tuned a sentiment classifier
  reaching **[your accuracy]%** accuracy on 5,000 held-out reviews.
- Diagnosed and fixed a NaN-propagation bug in attention masking for padded sequences.

## Layout

```
run_all.py                     one-command pipeline
src/model.py                   transformer architecture
src/pretrain_imdb_lm.py        Phase A
src/finetune_imdb_classifier.py  Phase B (prints accuracy/F1/ROC-AUC)
src/classifier_model.py        classification head + context extension
src/imdb_data.py               tokenizer / dataset loading
src/train.py, data.py, generate.py   Part 1 (Shakespeare)
app.py                         Streamlit demo (run after training)
extras/                        optional HuggingFace DistilBERT scripts (untested;
                               need `pip install transformers datasets` + internet)
```
