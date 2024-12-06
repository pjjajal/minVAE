"""This module contains an implementation of the Discrete Wavelet (Packet) Transform (DWT) and Inverse Discrete Wavelet Transform (IDWT).
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from ptwt.conv_transform_2 import wavedec2, waverec2
from pywt import Wavelet


class WaveletTransform(nn.Module):
    def __init__(
        self,
        in_channels=3,
        wavelet="haar",
        maxlevel=1,
        mode: Literal["zero", "reflect", "replicate", "circular"] = "replicate",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.maxlevel = maxlevel
        self.wavelet = wavelet
        self.mode = "constant" if mode == "zero" else mode
        self.leaf_nodes = int(4**maxlevel)
        self.out_channels = int(self.leaf_nodes * in_channels)

        dec_lo, dec_hi, rec_lo, rec_hi = Wavelet(wavelet).filter_bank
        self.register_buffer("dec_lo", torch.tensor(dec_lo, dtype=torch.float32))
        self.register_buffer("dec_hi", torch.tensor(dec_hi, dtype=torch.float32))
        self.register_buffer("rec_lo", torch.tensor(rec_lo, dtype=torch.float32))
        self.register_buffer("rec_hi", torch.tensor(rec_hi, dtype=torch.float32))

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.deconstruct(x)

    def deconstruct(self, x):
        for _ in range(self.maxlevel):
            a, h, v, d = self._dwt(x)
            x = torch.cat([a, h, v, d], dim=1)
        return x

    def _dwt(self, x):
        dtype = x.dtype
        device = x.device

        n = self.dec_lo.shape[0]
        g = x.shape[1]
        p = (n - 2) // 2

        # these are going to be shape (oC, C/g, n)
        hl = (
            self.dec_lo.flip(0)
            .reshape(1, 1, -1)
            .repeat(g, 1, 1)
            .to(dtype=dtype, device=device)
        )
        hh = (
            self.dec_hi.flip(0)
            .reshape(1, 1, -1)
            .repeat(g, 1, 1)
            .to(dtype=dtype, device=device)
        )

        x = F.pad(x, pad=(p, p, p, p), mode=self.mode).to(dtype=dtype)
        # the unsqueeze is done to (oC, C/g, 1, n)
        xl = F.conv2d(x, hl.unsqueeze(2), groups=g, stride=(1, 2))
        xh = F.conv2d(x, hh.unsqueeze(2), groups=g, stride=(1, 2))

        # the unsqueeze is done to (oC, C/g, n, 1)
        xll = F.conv2d(xl, hl.unsqueeze(3), groups=g, stride=(2, 1))
        xlh = F.conv2d(xl, hh.unsqueeze(3), groups=g, stride=(2, 1))
        xhl = F.conv2d(xh, hl.unsqueeze(3), groups=g, stride=(2, 1))
        xhh = F.conv2d(xh, hh.unsqueeze(3), groups=g, stride=(2, 1))

        return xll, xlh, xhl, xhh

    def reconstruct(self, x):
        assert x.shape[1] == int(
            self.leaf_nodes * self.in_channels
        ), f"Not enough channels for reconstruction at level {self.maxlevel}, expected {self.leaf_nodes * self.in_channels}, got {x.shape[1]}"

        for _ in range(self.maxlevel):
            x = self._idwt(x)
        return x

    def _idwt(self, coeffs):
        dtype = coeffs.dtype
        device = coeffs.device

        n = self.dec_lo.shape[0]
        g = coeffs.shape[1] // 4
        p = (n - 2) // 2

        # these are going to be shape (oC, C/g, n)
        hl = (
            self.rec_lo.reshape(1, 1, -1).repeat(g, 1, 1).to(dtype=dtype, device=device)
        )
        hh = (
            self.rec_hi.reshape(1, 1, -1).repeat(g, 1, 1).to(dtype=dtype, device=device)
        )

        xll, xlh, xhl, xhh = coeffs.chunk(4, dim=1)
        yl = F.conv_transpose2d(
            xll, hl.unsqueeze(3), groups=g, stride=(2, 1), padding=(p, 0)
        )
        yl += F.conv_transpose2d(
            xlh, hh.unsqueeze(3), groups=g, stride=(2, 1), padding=(p, 0)
        )
        yh = F.conv_transpose2d(
            xhl, hl.unsqueeze(3), groups=g, stride=(2, 1), padding=(p, 0)
        )
        yh += F.conv_transpose2d(
            xhh, hh.unsqueeze(3), groups=g, stride=(2, 1), padding=(p, 0)
        )
        y = F.conv_transpose2d(
            yl, hl.unsqueeze(2), groups=g, stride=(1, 2), padding=(0, p)
        )
        y += F.conv_transpose2d(
            yh, hh.unsqueeze(2), groups=g, stride=(1, 2), padding=(0, p)
        )
        return y


class _WaveletTransform(nn.Module):
    def __init__(self, in_channels=3, wavelet="haar", maxlevel=1, mode="zero"):
        super().__init__()
        self.in_channels = in_channels
        self.maxlevel = maxlevel
        self.wavelet = wavelet
        self.mode = mode
        self.leaf_nodes = int(
            4**maxlevel
        )  # the number of coefficients at the last level
        self.out_channels = int(self.leaf_nodes * in_channels)

    @torch.compiler.disable()
    def forward(self, x):
        return self.deconstruct(x)

    @torch.compiler.disable()
    def deconstruct(self, x):
        for _ in range(self.maxlevel):
            a, (h, v, d) = wavedec2(
                x,
                wavelet=self.wavelet,
                level=1,
                mode=self.mode,
            )
            x = torch.cat([a, h, v, d], dim=1)
        return x

    @torch.compiler.disable()
    def reconstruct(self, x):
        assert x.shape[1] == int(
            self.leaf_nodes * self.in_channels
        ), f"Not enough channels for reconstruction at level {self.maxlevel}, expected {self.leaf_nodes * self.in_channels}, got {x.shape[1]}"

        for _ in range(self.maxlevel):
            a, *coeffs = x.chunk(4, dim=1)
            x = waverec2((a, tuple(coeffs)), wavelet=self.wavelet)
        return x
