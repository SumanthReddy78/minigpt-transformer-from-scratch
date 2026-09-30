"""
Load a trained checkpoint and generate text from a prompt.

Run:
    python src/generate.py --prompt "ROMEO:" --tokens 200
"""

import argparse
import os
import torch

from model import MiniGPT

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoint.pt")


def load_model(checkpoint_path: str = CHECKPOINT_PATH, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = MiniGPT(
        vocab_size=cfg["vocab_size"],
        block_size=cfg["block_size"],
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    stoi, itos = ckpt["stoi"], ckpt["itos"]
    return model, stoi, itos


def generate_text(model, stoi, itos, prompt: str, max_new_tokens: int = 200,
                   temperature: float = 0.8, top_k: int = 40, device: str = "cpu"):
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return "".join(itos[i] for i in out[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="ROMEO:")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    args = parser.parse_args()

    model, stoi, itos = load_model(args.checkpoint)
    text = generate_text(model, stoi, itos, args.prompt, args.tokens)
    print(text)
