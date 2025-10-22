# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed

from .layers import (
    TimestepEmbedder,
    LabelEmbedder,
    DiTBlock,
    FinalLayer,
    get_2d_sincos_pos_embed
)


"""simple mlp layers"""
def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )


"""SiT models."""
class SiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        use_cfg_embedding=True,
        num_classes=1000,
        repa_depth=0,
        z_dims=[768],
        projector_dim=2048,
        **block_kwargs # fused_attn
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.repa_depth = repa_depth
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.num_patches = (input_size // patch_size)**2
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, use_cfg_embedding)
        
        if self.repa_depth > 0:
            self.projectors = nn.ModuleList([
                build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
            ])

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
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

    def unpatchify(self, x):
        c, p = self.out_channels, self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, y, return_feat=False):
        t = t.reshape(-1)
        x = self.x_embedder(x) + self.pos_embed  # (B, L, D)
        B, L, D = x.shape
        t_embed = self.t_embedder(t).reshape(B, -1, D)
        y_embed = self.y_embedder(y).reshape(B, -1, D)
        c = t_embed + y_embed
        for l, block in enumerate(self.blocks):
            x = block(x, c)
            if self.repa_depth > 0 and (l + 1) == self.repa_depth:
                zs = [projector(x.reshape(-1, D)).reshape(B, L, -1) for projector in self.projectors]
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        if self.repa_depth > 0 and return_feat:
            return x, zs
        return x
    

SiT_XL_2 = partial(SiT, depth=28, hidden_size=1152, patch_size=2, num_heads=16)
SiT_XL_4 = partial(SiT, depth=28, hidden_size=1152, patch_size=4, num_heads=16)
SiT_XL_8 = partial(SiT, depth=28, hidden_size=1152, patch_size=8, num_heads=16)
SiT_L_2  = partial(SiT, depth=24, hidden_size=1024, patch_size=2, num_heads=16)
SiT_L_4  = partial(SiT, depth=24, hidden_size=1024, patch_size=4, num_heads=16)
SiT_L_8  = partial(SiT, depth=24, hidden_size=1024, patch_size=8, num_heads=16)
SiT_M_2  = partial(SiT, depth=16, hidden_size=1024, patch_size=2, num_heads=16)
SiT_M_4  = partial(SiT, depth=16, hidden_size=1024, patch_size=4, num_heads=16)
SiT_M_8  = partial(SiT, depth=16, hidden_size=1024, patch_size=8, num_heads=16)
SiT_B_2  = partial(SiT, depth=12, hidden_size=768, patch_size=2, num_heads=12)
SiT_B_4  = partial(SiT, depth=12, hidden_size=768, patch_size=4, num_heads=12)
SiT_B_8  = partial(SiT, depth=12, hidden_size=768, patch_size=8, num_heads=12)
SiT_S_2  = partial(SiT, depth=12, hidden_size=384, patch_size=2, num_heads=6)
SiT_S_4  = partial(SiT, depth=12, hidden_size=384, patch_size=4, num_heads=6)
SiT_S_8  = partial(SiT, depth=12, hidden_size=384, patch_size=8, num_heads=6)


SiT_models = {
    "SiT-XL/2": SiT_XL_2,  "SiT-XL/4": SiT_XL_4,  "SiT-XL/8": SiT_XL_8,
    "SiT-L/2":  SiT_L_2,   "SiT-L/4":  SiT_L_4,   "SiT-L/8":  SiT_L_8,
    "SiT-B/2":  SiT_B_2,   "SiT-B/4":  SiT_B_4,   "SiT-B/8":  SiT_B_8,
    "SiT-S/2":  SiT_S_2,   "SiT-S/4":  SiT_S_4,   "SiT-S/8":  SiT_S_8,
}


if __name__ == "__main__":
    # Test the model initialization
    model = SiT_models["SiT-XL/2"](input_size=32, in_channels=4, num_classes=1000)
    print(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    B, C, H, W = 4, 4, 32, 32
    x = torch.randn((B, C, H, W)).to(device=device, dtype=dtype)
    t = torch.rand((B, )).to(device=device, dtype=dtype)
    y = torch.randint(0, 1000, (B,)).to(device=device)
    output = model(x, t, y)
    print(output.shape)
