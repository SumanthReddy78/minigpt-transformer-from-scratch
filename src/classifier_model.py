"""
Wraps the pretrained MiniGPT backbone with a classification head.

Pooling: mean over all real (non-padded) token positions, masked so padding
never contributes. Last-token-only pooling (the original GPT-1 recipe) was
tried first — it only requires the final position to encode the whole
review, which is a lot to ask of one vector. Mean pooling spreads that
load across every real token's representation and empirically does better
for review-length sentiment classification.
"""

import torch
import torch.nn as nn

from model import MiniGPT


class TransformerClassifier(nn.Module):
    def __init__(self, backbone: MiniGPT, embed_dim: int, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier_head = nn.Linear(embed_dim, num_classes)

    def forward(self, idx: torch.Tensor, attention_mask: torch.Tensor):
        hidden = self.backbone.encode(idx, attention_mask=attention_mask)  # (B, T, E)

        mask = attention_mask.unsqueeze(-1)  # (B, T, 1)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)  # avoid divide-by-zero
        pooled = summed / counts

        pooled = self.dropout(pooled)
        return self.classifier_head(pooled)

    @classmethod
    def from_pretrained_checkpoint(cls, checkpoint_path: str, num_classes: int = 2,
                                    dropout: float = 0.1, device: str = "cpu",
                                    target_block_size: int = None):
        """
        target_block_size lets the classifier use longer sequences than the
        backbone was pretrained with. Every backbone weight except the
        position-embedding table is independent of sequence length (attention
        and FFN weights are applied per-token, regardless of how many tokens
        there are), so only that table needs to change size. New position
        slots are initialized by tiling the pretrained table — a reasonable
        starting point that fine-tuning then adapts — rather than redoing the
        entire (expensive) language-model pretraining phase from scratch.
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        orig_block_size = cfg["block_size"]
        new_block_size = target_block_size or orig_block_size

        backbone = MiniGPT(
            vocab_size=cfg["vocab_size"],
            block_size=new_block_size,
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
        )

        state_dict = dict(ckpt["model_state"])
        if new_block_size != orig_block_size:
            old_pos = state_dict.pop("pos_emb.weight")  # (orig_block_size, E)
            reps = (new_block_size + orig_block_size - 1) // orig_block_size
            tiled = old_pos.repeat(reps, 1)[:new_block_size]
            backbone.load_state_dict(state_dict, strict=False)
            with torch.no_grad():
                backbone.pos_emb.weight.copy_(tiled)
        else:
            backbone.load_state_dict(state_dict)

        model = cls(backbone, embed_dim=cfg["embed_dim"], num_classes=num_classes, dropout=dropout).to(device)
        cfg = {**cfg, "block_size": new_block_size}
        return model, ckpt["vocab_stoi"], ckpt["vocab_itos"], cfg
