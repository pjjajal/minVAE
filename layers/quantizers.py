import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, pack, unpack


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


class FSQuantizer(nn.Module):
    def __init__(
        self,
        levels: list[int],
        input_dim: int,
        num_codebooks: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.dtype = torch.float32
        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels)

        _basis = torch.cumprod(
            torch.tensor([1] + levels[:-1]),
            dim=0,
            dtype=torch.int32,
        )
        self.register_buffer("_basis", _basis)

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        self.dim = input_dim
        self.proj_in = nn.Linear(input_dim, effective_codebook_dim)
        self.proj_out = nn.Linear(effective_codebook_dim, input_dim)

        self.codebook_size = self._levels.prod().item()

    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: torch.Tensor) -> torch.Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat).float()
        return (zhat * self._basis).sum(dim=-1).to(torch.int32)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Inverse of `codes_to_indices`."""
        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        codes = rearrange(codes, "... c d -> ... (c d)")
        codes = self.proj_out(codes)

        codes = rearrange(codes, "b ... d -> b d ...")

        return codes.to(self.dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # move channels to last dimension.
        z = rearrange(z, "b d ... -> b ... d")
        z, ps = pack_one(z, "b * d")

        assert z.shape[-1] == self.dim, f"expected input dim {self.dim}, got {z.shape[-1]}"

        z = self.proj_in(z)

        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, "b n c d -> b n (c d)")

        out = self.proj_out(codes)

        out = unpack_one(out, ps, "b * d")
        out = rearrange(out, "b ... d -> b d ...")
        indices = unpack_one(indices, ps, "b * c")

        return out, indices