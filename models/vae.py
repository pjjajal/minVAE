import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.distributions import GaussianDistribution
from layers.layers2d import Encoder, Decoder
from layers.wavelet import WaveletTransform, IdentityTransform


class VAE(nn.Module):
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
        wavelet: str = None,
        maxlevel: int = 1,
    ):
        super().__init__()
        self.z_channels = z_channels
        self.spatial_compression = spatial_compression

        # Wavelet Transform
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
            z_channels=int(2 * z_channels), # double channels for mean and variance
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
        self.distribution = GaussianDistribution()

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.distribution(h)

    def decode(self, z: torch.Tensor):
        dec = self.decoder(z)
        return dec

    def forward(self, x: torch.Tensor):
        x = self.wavelet_transform(x)
        z, posteriors = self.encode(x)
        reconstruction = self.decode(z)
        reconstruction = self.wavelet_transform.reconstruct(reconstruction)
        return reconstruction, posteriors
