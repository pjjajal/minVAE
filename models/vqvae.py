import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.layers2d import Encoder, Decoder
from layers.wavelet import WaveletTransform, IdentityTransform
from layers.quantizers import FSQuantizer


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        resolution: int,
        z_channels: int,
        spatial_compression: int,
        quantization: str,
        levels: list[int],
        embedding_dim: int,
        num_codebooks: int = 1,
        rotate: bool = False,
        wavelet: str = None,
        maxlevel: int = 1,
    ):
        super().__init__()
        self.z_channels = z_channels
        self.spatial_compression = spatial_compression

        # Wavelet Transform
        print(f"Wavelet Transform: {wavelet}")
        self.wavelet_transform = (
            WaveletTransform(
                in_channels=in_channels,
                wavelet=wavelet,
                maxlevel=maxlevel,
            )
            if wavelet
            else IdentityTransform()
        )

        # Update in_channels, out_channels, and resolution when using Wavelet Transform
        self.in_channels = (
            self.wavelet_transform.out_channels if wavelet else in_channels
        )
        self.out_channels = (
            self.wavelet_transform.out_channels if wavelet else out_channels
        )
        self.resolution = resolution // 2**maxlevel if wavelet else resolution
        # Encoder and Decoder
        self.encoder = Encoder(
            in_channels=self.in_channels,
            channels=channels,
            channels_mult=channels_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resolution=self.resolution,
            z_channels=z_channels,
            spatial_compression=spatial_compression,
        )
        self.decoder = Decoder(
            z_channels=z_channels,
            channels=channels,
            channels_mult=channels_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resolution=self.resolution,
            out_channels=self.out_channels,
            spatial_compression=spatial_compression,
        )
        effective_embedding_dim = num_codebooks * embedding_dim
        self.quant_conv = nn.Conv2d(z_channels, effective_embedding_dim, 1)
        self.post_quant_conv = nn.Conv2d(effective_embedding_dim, z_channels, 1)
        if quantization == "fsq":
            self.quantizer = FSQuantizer(
                levels=levels,
                input_dim=effective_embedding_dim,
                num_codebooks=num_codebooks,
                rotate=rotate,
            )

    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.quant_conv(x)
        codes, indices = self.quantizer(x)
        return codes, indices

    def decode(self, z: torch.Tensor):
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        x = self.wavelet_transform(x)
        codes, indices = self.encode(x)
        reconstruction = self.decode(codes)
        reconstruction = self.wavelet_transform.reconstruct(reconstruction)
        return reconstruction, (indices,)
