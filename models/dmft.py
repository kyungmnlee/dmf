# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

from functools import partial

import torch
from torch import Tensor
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed
from .layers import (
    TimestepEmbedder,
    LabelEmbedder,
    DiTBlock,
    FinalLayer,
    get_2d_sincos_pos_embed,
    timestep_embedding
)


"""Decoupled MeanFlow Transformer (DMFT)"""
class DMFT(nn.Module):
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_cfg_embedding: bool = True,
        num_classes: int = 1000,
        dmf_depth: int = 8,
        use_logvar: bool = False,
        **block_kwargs # fused_attn, qk_norm, post_ln
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.dmf_depth = dmf_depth
        self.use_logvar = use_logvar

        # model
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, use_cfg_embedding)
        # Will use fixed sin-cos embedding:
        self.num_patches = self.x_embedder.num_patches
        # (input_size // patch_size)**2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size), requires_grad=False
        )
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        if self.use_logvar:
            self.logvar_t_embedder = partial(timestep_embedding, dim=128)
            self.logvar_linear  = nn.Linear(256, 1)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
        if self.use_logvar:
            nn.init.constant_(self.logvar_linear.weight, 0)
            nn.init.constant_(self.logvar_linear.bias, 0)

    def unpatchify(self, x: Tensor) -> Tensor:
        c, p = self.out_channels, self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, r, y, return_logvar: bool = False):
        t, r = t.reshape(-1), r.reshape(-1)
        x = self.x_embedder(x) + self.pos_embed  # (B, L, D)
        B, L, D = x.shape
        t_embed = self.t_embedder(t).reshape(B, -1, D)
        y_embed = self.y_embedder(y).reshape(B, -1, D)
        c_t = t_embed + y_embed
        
        # Encoder blocks
        for _, block in enumerate(self.blocks[:self.dmf_depth]):
            x = block(x, c_t)
        # Decoder blocks
        r_embed = self.t_embedder(r).reshape(B, -1, D)
        c_r = r_embed + y_embed
        for _, block in enumerate(self.blocks[self.dmf_depth:]):
            x = block(x, c_r)
        x = self.final_layer(x, c_r)
        x = self.unpatchify(x)
        # logvar = None
        if return_logvar:
            logvar = self.logvar_linear(
                torch.cat([self.logvar_t_embedder(t), self.logvar_t_embedder(r)], dim=1)
            )
            logvar = logvar.view(-1, *[1] * (x.ndim - 1))
            return x, logvar
        return x


DMFT_XL_2 = partial(DMFT, depth=28, hidden_size=1152, patch_size=2, num_heads=16)
DMFT_XL_4 = partial(DMFT, depth=28, hidden_size=1152, patch_size=4, num_heads=16)
DMFT_XL_8 = partial(DMFT, depth=28, hidden_size=1152, patch_size=8, num_heads=16)
DMFT_L_2  = partial(DMFT, depth=24, hidden_size=1024, patch_size=2, num_heads=16)
DMFT_L_4  = partial(DMFT, depth=24, hidden_size=1024, patch_size=4, num_heads=16)
DMFT_L_8  = partial(DMFT, depth=24, hidden_size=1024, patch_size=8, num_heads=16)
DMFT_M_2  = partial(DMFT, depth=16, hidden_size=1024, patch_size=2, num_heads=16)
DMFT_M_4  = partial(DMFT, depth=16, hidden_size=1024, patch_size=4, num_heads=16)
DMFT_M_8  = partial(DMFT, depth=16, hidden_size=1024, patch_size=8, num_heads=16)
DMFT_B_2  = partial(DMFT, depth=12, hidden_size=768, patch_size=2, num_heads=12)
DMFT_B_4  = partial(DMFT, depth=12, hidden_size=768, patch_size=4, num_heads=12)
DMFT_B_8  = partial(DMFT, depth=12, hidden_size=768, patch_size=8, num_heads=12)
DMFT_S_2  = partial(DMFT, depth=12, hidden_size=384, patch_size=2, num_heads=6)
DMFT_S_4  = partial(DMFT, depth=12, hidden_size=384, patch_size=4, num_heads=6)
DMFT_S_8  = partial(DMFT, depth=12, hidden_size=384, patch_size=8, num_heads=6)

DMFT_models = {
    "DMFT-XL/2": DMFT_XL_2,  "DMFT-XL/4": DMFT_XL_4,  "DMFT-XL/8": DMFT_XL_8,
    "DMFT-L/2":  DMFT_L_2,   "DMFT-L/4":  DMFT_L_4,   "DMFT-L/8":  DMFT_L_8,
    "DMFT-B/2":  DMFT_B_2,   "DMFT-B/4":  DMFT_B_4,   "DMFT-B/8":  DMFT_B_8,
    "DMFT-S/2":  DMFT_S_2,   "DMFT-S/4":  DMFT_S_4,   "DMFT-S/8":  DMFT_S_8,
}

