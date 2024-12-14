import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, pack, unpack


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def get_very_efficient_rotation(u, q, e):
    e = rearrange(e, "b d -> b 1 d")
    w = ((u + q) / torch.norm(u + q, dim=1, keepdim=True)).detach()
    e = (
        e
        - 2 * (e @ rearrange(w, "b d -> b d 1") @ rearrange(w, "b d -> b 1 d"))
        + 2
        * (
            e
            @ rearrange(u, "b d -> b d 1").detach()
            @ rearrange(q, "b d -> b 1 d").detach()
        )
    )
    return e


def round_rot(z: torch.Tensor) -> torch.Tensor:
    """Round with rotation-trick."""
    b, l, c, d = z.shape
    e = z
    q = e.round()

    # rearrage
    e, ps = pack_one(e, "* d")
    q, _ = pack_one(q, "* d")
    lam = (q.norm(dim=1, keepdim=True) / e.norm(dim=1, keepdim=True)).detach()

    # rotation trick
    q = (
        lam
        * get_very_efficient_rotation(
            e / (torch.norm(e, dim=1, keepdim=True) + 1e-6),
            q / (torch.norm(q, dim=1, keepdim=True) + 1e-6),
            e,
        ).squeeze()
    )
    q = unpack_one(q, ps, "* d")
    return q


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


class FSQuantizer(nn.Module):
    def __init__(
        self,
        levels: list[int],
        input_dim: int,
        num_codebooks: int = 1,
        rotate: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.rotate = rotate  # use rotation trick
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

        self.keep_num_codebooks_dim = num_codebooks > 1

        self.dim = input_dim
        has_projections = self.dim != effective_codebook_dim
        self.proj_in = (
            nn.Linear(input_dim, effective_codebook_dim)
            if has_projections
            else nn.Identity()
        )
        self.proj_out = (
            nn.Linear(effective_codebook_dim, input_dim)
            if has_projections
            else nn.Identity()
        )

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(
            torch.arange(self.codebook_size), proj_out=False
        )
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        if self.rotate:
            quantized = round_rot(self.bound(z))
        else:
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

    def indices_to_codes(self, indices: torch.Tensor, proj_out=False) -> torch.Tensor:
        """Inverse of `codes_to_indices`."""
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        if proj_out:
            codes = self.proj_out(codes)

        if is_img_or_video:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes.to(self.dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # move channels to last dimension.
        z = rearrange(z, "b d ... -> b ... d")
        z, ps = pack_one(z, "b * d")

        assert (
            z.shape[-1] == self.dim
        ), f"expected input dim {self.dim}, got {z.shape[-1]}"

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
