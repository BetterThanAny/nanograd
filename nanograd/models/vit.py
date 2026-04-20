"""Vision Transformer (Dosovitskiy et al. 2020) for small images."""
from __future__ import annotations

from typing import Optional

import numpy as np

from nanograd.nn.attention import LearnedPositionalEncoding, TransformerBlock
from nanograd.nn.conv import Conv2d
from nanograd.nn.layers import LayerNorm, Linear
from nanograd.nn.module import Module, Parameter
from nanograd.tensor import Tensor


class PatchEmbed(Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        seed: Optional[int] = None,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, seed=seed
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W) -> (B, E, H/p, W/p) -> (B, N, E)
        x = self.proj(x)
        B, E, Hp, Wp = x.shape
        return x.reshape(B, E, Hp * Wp).transpose(0, 2, 1)


class ViT(Module):
    def __init__(
        self,
        image_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        num_classes: int = 10,
        embed_dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        ff_dim: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        s = seed or 0
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim, seed=s)
        N = self.patch_embed.num_patches
        rng = np.random.default_rng(s + 1)
        self.cls_token = Parameter(rng.standard_normal((1, 1, embed_dim)).astype(np.float32) * 0.02)
        self.pos_enc = LearnedPositionalEncoding(N + 1, embed_dim, seed=s + 2)
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim=ff_dim, seed=s + 10 + 10 * i)
            for i in range(depth)
        ]
        for i, blk in enumerate(self.blocks):
            setattr(self, f"block{i}", blk)
        self.ln = LayerNorm(embed_dim)
        self.head = Linear(embed_dim, num_classes, seed=s + 1000)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)  # (B, N, E)
        B = x.shape[0]
        # expand CLS token to batch via broadcasting + cat
        cls = self.cls_token.expand(B, 1, self.cls_token.shape[-1])
        from nanograd import cat as _cat
        x = _cat([cls, x], axis=1)  # (B, N+1, E)
        x = self.pos_enc(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        return self.head(x[:, 0])  # CLS token head
