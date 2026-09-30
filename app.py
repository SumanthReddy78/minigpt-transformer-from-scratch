"""
Streamlit demo for the ML Engineer transformer project.

Two tabs, both backed by models actually trained in this project:
1. Text generation from the from-scratch MiniGPT (checkpoint.pt)
2. Sentiment classification from the same architecture, pretrained via
   language modeling on real IMDB text then fine-tuned with a
   classification head (imdb_backbone.pt + imdb_classifier.pt)

Run:
    streamlit run app.py
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

st.set_page_config(page_title="Transformer Portfolio Project", layout="centered")
st.title("Transformer From Scratch + Fine-Tuned Classifier")
st.caption("ML Engineer portfolio project — architecture, pretraining, and fine-tuning all built and trained end to end.")

tab1, tab2 = st.tabs(["Text Generation (from scratch)", "Sentiment Classifier (pretrained + fine-tuned)"])

with tab1:
    st.subheader("Generate text with the from-scratch MiniGPT")
    checkpoint_path = "checkpoint.pt"

    if not os.path.exists(checkpoint_path):
        st.warning("No checkpoint found yet. Run `python src/train.py --target_total 2000` first.")
    else:
        from generate import load_model, generate_text

        prompt = st.text_input("Prompt", value="ROMEO:")
        max_tokens = st.slider("Tokens to generate", 20, 500, 200)
        temperature = st.slider("Temperature", 0.1, 1.5, 0.8)

        if st.button("Generate"):
            model, stoi, itos = load_model(checkpoint_path)
            with st.spinner("Generating..."):
                output = generate_text(model, stoi, itos, prompt, max_tokens, temperature)
            st.text_area("Output", output, height=250)

with tab2:
    st.subheader("Classify sentiment with the fine-tuned model")
    backbone_path = "imdb_backbone.pt"
    classifier_path = "imdb_classifier.pt"

    if not os.path.exists(classifier_path):
        st.warning(
            "No fine-tuned classifier found yet. Run "
            "`python src/pretrain_imdb_lm.py --target_total 2000` then "
            "`python src/finetune_imdb_classifier.py --target_total 600` first."
        )
    else:
        import torch
        from classifier_model import TransformerClassifier
        from imdb_data import Vocab, tokenize

        @st.cache_resource
        def load_classifier():
            ckpt = torch.load(classifier_path, map_location="cpu", weights_only=False)
            cfg = ckpt["backbone_config"]
            from model import MiniGPT

            backbone = MiniGPT(
                vocab_size=cfg["vocab_size"], block_size=cfg["block_size"],
                embed_dim=cfg["embed_dim"], num_heads=cfg["num_heads"], num_layers=cfg["num_layers"],
            )
            model = TransformerClassifier(backbone, embed_dim=cfg["embed_dim"])
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            return model, ckpt["vocab_stoi"], cfg["block_size"]

        def encode_for_inference(text, stoi, max_len):
            ids = [stoi.get(w, stoi["<unk>"]) for w in tokenize(text)]
            true_len = min(len(ids), max_len)
            ids = ids[:max_len]
            ids = [stoi["<pad>"]] * (max_len - len(ids)) + ids
            return ids, true_len

        text = st.text_area("Enter a movie review", "This film was a genuine surprise — sharp writing, great pacing.")

        if st.button("Classify"):
            model, stoi, max_len = load_classifier()
            ids, true_len = encode_for_inference(text, stoi, max_len)
            ids_t = torch.tensor([ids], dtype=torch.long)
            mask_t = torch.zeros(1, max_len)
            mask_t[0, max_len - true_len:] = 1.0

            with torch.no_grad():
                logits = model(ids_t, mask_t)
                probs = torch.softmax(logits, dim=1)[0]

            label = "Positive" if probs[1] > probs[0] else "Negative"
            st.metric("Prediction", label, f"{max(probs).item():.1%} confidence")

        st.caption("Test-set performance: 84.6% accuracy · 84.7% F1 · 0.925 ROC-AUC (5,000 held-out reviews, 250-word context)")
