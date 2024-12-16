"""This module contains 2D layers for the VAE.

This is adapted from CosmosTokenizer (which is adapted from StableDiffusionVAE).


Links:
- CosmosTokenizer: https://github.com/NVIDIA/Cosmos-Tokenizer/blob/e166d17d8a1e3a761635d7c100d556d2a434ee48/cosmos_tokenizer/modules/layers2d.py
- StableDiffusion: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def get_block_fn(block_type: str):
    if block_type == "resnet":
        return ResnetBlock
    elif block_type == "convnext":
        return ConvNeXtBlock
    else:
        raise ValueError(f"block type {block_type} not supported.")


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups, in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtBlock(nn.Module):
    def __init__(
        self, *, in_channels: int, out_channels: int = None, dropout: float, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        self.convdw1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=7,
            padding=3,
            groups=in_channels,
        )
        self.norm1 = nn.LayerNorm(in_channels)
        self.pwconv1_1 = nn.Linear(in_channels, 4 * in_channels)
        self.act1 = nn.GELU()
        self.gn1 = GRN(4 * in_channels)
        self.pwconv1_2 = nn.Linear(4 * in_channels, in_channels)

        self.up_proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.nin_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        h = x
        h = self.convdw1(h)
        h = rearrange(h, "b c h w -> b h w c")
        h = self.norm1(h)
        h = self.pwconv1_1(h)
        h = self.act1(h)
        h = self.gn1(h)
        h = self.pwconv1_2(h)
        h = rearrange(h, "b h w c -> b c h w")

        x = self.up_proj(h) + self.nin_shortcut(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self, *, in_channels: int, out_channels: int = None, dropout: float, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, self.out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = Normalize(self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1
        )
        self.nin_shortcut = (
            nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1)
            if in_channels != self.out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.head_dim = 64
        self.num_heads = in_channels // self.head_dim

        self.norm = Normalize(in_channels)
        self.qkv = nn.Conv2d(
            in_channels, 3 * in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        qkv = self.qkv(h_)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        B, C, H, W = q.shape
        q = rearrange(q, "b (h d) x y -> b h (x y) d", h=self.num_heads)
        k = rearrange(k, "b (h d) x y -> b h (x y) d", h=self.num_heads)
        v = rearrange(v, "b (h d) x y -> b h (x y) d", h=self.num_heads)
        h_ = F.scaled_dot_product_attention(q, k, v)
        h_ = rearrange(h_, "b h (x y) d -> b (h d) x y", x=H, y=W)
        h_ = self.proj_out(h_)
        return x + h_


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        resolution: int,
        z_channels: int,
        spatial_compression: int,
        block_fn: nn.Module = ResnetBlock,
    ):
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks

        # number of downsamples.
        assert spatial_compression % 2 == 0, "spatial compression must be a power of 2."
        self.num_downsamples = int(math.log2(spatial_compression))
        assert (
            self.num_downsamples <= self.num_resolutions
        ), f"we can only at most downsample {self.num_resolutions} times."

        # input projection (embedding).
        in_channels = in_channels
        self.conv_in = nn.Conv2d(
            in_channels, channels, kernel_size=3, padding=1, stride=1
        )

        # downsample.
        curr_resolution = resolution
        in_ch_mult = (1,) + tuple(channels_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = channels * in_ch_mult[i_level]
            block_out = channels * channels_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    block_fn(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_resolution in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level < self.num_downsamples:
                down.downsample = Downsample(block_in)
                curr_resolution //= 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block1 = block_fn(
            in_channels=block_in, out_channels=block_in, dropout=dropout
        )
        self.mid.attn1 = AttnBlock(block_in)
        self.mid.block2 = block_fn(
            in_channels=block_in, out_channels=block_in, dropout=dropout
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in, z_channels, kernel_size=3, padding=1, stride=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # downsample.
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level < self.num_downsamples:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block1(h)
        h = self.mid.attn1(h)
        h = self.mid.block2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        resolution: int,
        out_channels: int,
        spatial_compression: int,
        block_fn: nn.Module = ResnetBlock,
    ):
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks

        # number of upsamples.
        assert spatial_compression % 2 == 0, "spatial compression must be a power of 2."
        self.num_upsamples = int(math.log2(spatial_compression))
        assert (
            self.num_upsamples <= self.num_resolutions
        ), f"we can only at most upsample {self.num_resolutions} times."

        # input projection (embedding).
        block_in = channels * channels_mult[self.num_resolutions - 1]
        curr_resolution = resolution // spatial_compression
        self.z_shape = (1, z_channels, curr_resolution, curr_resolution)
        print(f"z of shape: {self.z_shape}, dimensions: {np.prod(self.z_shape)}")

        # input projection (embedding).
        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=3, padding=1, stride=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block1 = block_fn(
            in_channels=block_in, out_channels=block_in, dropout=dropout
        )
        self.mid.attn1 = AttnBlock(block_in)
        self.mid.block2 = block_fn(
            in_channels=block_in, out_channels=block_in, dropout=dropout
        )

        # upsample.
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = channels * channels_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    block_fn(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_resolution in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level >= (self.num_resolutions - self.num_upsamples):
                up.upsample = Upsample(block_in)
                curr_resolution *= 2
            self.up.append(up)

        # output projection.
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in, out_channels, kernel_size=3, padding=1, stride=1
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # input projection.
        h = self.conv_in(z)

        # middle
        h = self.mid.block1(h)
        h = self.mid.attn1(h)
        h = self.mid.block2(h)

        # upsample.
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level < self.num_upsamples:
                h = self.up[i_level].upsample(h)

        # output projection.
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
