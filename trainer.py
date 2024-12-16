import random
from datetime import datetime
from functools import partial
from pathlib import Path

import hydra
import lightning as L
import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.distributed import split_dataset_by_node
from dreamsim import dreamsim
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import grad_norm
from omegaconf import DictConfig, OmegaConf
from torchmetrics.image import (
    FrechetInceptionDistance,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchvision.utils import make_grid

import helpers.loss as losses
import helpers.schedulers as schedulers
from dataset import create_dataset
from dataset.transforms import base_train_transform, val_transform
from models.gan import create_discriminator
from models.vae import VAE
from models.vqvae import VQVAE

try:
    from distributed_shampoo.distributed_shampoo import DistributedShampoo
    from distributed_shampoo.shampoo_types import AdamGraftingConfig

    SHAMPOO_AVAILABLE = True
except ImportError:
    print("Distributed Shampoo not available")
    SHAMPOO_AVAILABLE = False

DEFAULT_CHECKPOINTS_PATH = Path("./checkpoints")

torch.set_float32_matmul_precision("medium")


def setup_checkpoint_dir(cfg: DictConfig):
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S.%f")
    ### In case we want to add a more descriptive checkpoint dir folder
    if cfg.trainer.checkpoint_name:
        now = (
            now + "-" + cfg.trainer.checkpoint_name.replace(" ", "-").replace("_", "-")
        )
    save_base_path = (
        cfg.trainer.checkpoint_save_dir
        if cfg.trainer.checkpoint_save_dir
        else DEFAULT_CHECKPOINTS_PATH
    )
    save_loc = cfg.trainer.save_loc if cfg.trainer.save_loc else save_base_path / now
    print(f"Saving checkpoints to: {save_loc}")
    return save_loc


def flatten_dict(d: DictConfig):
    out = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            for k2, v2 in flatten_dict(v).items():
                out[k + "." + k2] = v2
        else:
            out[k] = v
    return out


class VAEModel(L.LightningModule):
    def __init__(self, cfg: DictConfig, model: VAE):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.vae_cfg = cfg.model.vae_type
        self.is_discrete = self.vae_cfg.type == "discrete"
        self.logging_gradnorm = cfg.trainer.gradnorm_logging

        # manual optimization.
        self.automatic_optimization = False

        # which losses to use
        self.use_adversarial_loss = cfg.loss.adversarial_loss.enable
        self.use_perceptual_loss = cfg.loss.perceptual_loss.enable

        self.discriminator_warmup = cfg.optimizer.discriminator.discriminator_warmup

        # Loss weights
        self.kl_weight = cfg.loss.kl_weight
        self.perceptual_weight = cfg.loss.perceptual_loss.weight
        self.adversarial_weight = cfg.loss.adversarial_loss.weight

        # VAE losses
        self.reconstruction_loss = losses.reconstruction_loss(cfg.loss.recon_loss)
        self.kl_loss = losses.reg_loss

        # Adversarial loss
        if self.use_adversarial_loss:
            self.adverasarial_loss_type = cfg.loss.adversarial_loss.loss_type
            self.adverasarial_loss = partial(
                losses.discriminator_loss, type=cfg.loss.adversarial_loss.loss_type
            )
            self.generator_loss = partial(
                losses.generator_loss, type=cfg.loss.adversarial_loss.loss_type
            )
            self.discriminator = create_discriminator()

        # Perceptual loss
        self.preprocess = None
        if self.use_perceptual_loss:
            self.perceptual_loss_type = cfg.loss.perceptual_loss.type
            if cfg.loss.perceptual_loss.type == "lpips":
                self.perceptual_loss = lpips.LPIPS(
                    net_type=cfg.loss.perceptual_loss.model_name
                )
            elif cfg.loss.perceptual_loss.type == "dreamsim":
                self.perceptual_loss, self.preprocess = dreamsim(
                    dreamsim_type=cfg.loss.perceptual_loss.model_name,
                    pretrained=True,
                    cache_dir=cfg.loss.perceptual_loss.dreamsim_cache,
                )

        # EMA Model
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg.optimizer.ema),
        )

        # metrics
        self.psnr = PeakSignalNoiseRatio()
        self.ssim = StructuralSimilarityIndexMeasure()
        self.ema_psnr = PeakSignalNoiseRatio()
        self.ema_ssim = StructuralSimilarityIndexMeasure()

    def get_layer_layer(self):
        return self.model.decoder.conv_out

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        if self.use_adversarial_loss:
            vae_opt, discriminator_opt = optimizers
        else:
            vae_opt = optimizers
        vae_schedule = self.lr_schedulers()

        x, label = batch

        # forward pass
        reconstruction, posteriors = self.model.forward(x)
        if self.is_discrete:
            (indices,) = posteriors
        else:
            mu, logvar = posteriors

        # optimize discriminator
        d_loss = 0.0
        if self.use_adversarial_loss:
            adversarial_weight = (
                self.adversarial_weight
                if self.global_step > self.discriminator_warmup
                else 0.0
            )
            d_pred_real = self.discriminator(x)
            d_pred = self.discriminator(reconstruction.contiguous().detach())
            d_loss = self.adversarial_weight * self.adverasarial_loss(
                d_pred_real, d_pred
            )
            discriminator_opt.zero_grad()
            self.manual_backward(d_loss)
            discriminator_opt.step()

        # optimize vae
        reconstruction_loss = self.reconstruction_loss(reconstruction, x)

        kl_loss = 0.0
        if not self.is_discrete and self.vae_cfg.latent != "identity":
            kl_loss = self.kl_weight * self.kl_loss(mu, logvar)

        vae_loss = reconstruction_loss + kl_loss

        g_loss = 0.0
        if self.use_adversarial_loss:
            adversarial_weight = (
                self.adversarial_weight
                if self.global_step > self.discriminator_warmup
                else 0.0
            )
            d_pred = self.discriminator(reconstruction.contiguous())
            g_loss = self.generator_loss(d_pred)
            g_loss = (
                adversarial_weight
                * losses.calculate_adaptive_weight(
                    reconstruction_loss, g_loss, self.get_layer_layer()
                )
                * g_loss
            )
            vae_loss += g_loss
        perceptual_loss = 0.0
        if self.use_perceptual_loss:
            perceptual_loss = self.perceptual_loss(reconstruction.clamp(-1, 1), x)
            perceptual_loss = self.perceptual_weight * perceptual_loss.sum()
            vae_loss += perceptual_loss

        vae_opt.zero_grad()
        self.manual_backward(vae_loss)
        vae_opt.step()
        vae_schedule.step()

        # update EMA model
        self.ema_model.update_parameters(self.model)

        # log gradnorm
        if self.logging_gradnorm:
            norms = grad_norm(self.model, norm_type=2)
            self.log_dict(norms, sync_dist=True)

        # log losses
        self.log_dict(
            {
                "vae_loss": vae_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
                "perceptual_loss": perceptual_loss,
                "d_loss": d_loss,
                "g_loss": g_loss,
            },
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )

    def validation_step(self, batch, batch_idx):
        x, label = batch
        reconstruction, posteriors = self.model.forward(x)
        if self.is_discrete:
            (indices,) = posteriors
        else:
            mu, logvar = posteriors

        ema_reconstruction, ema_posteriors = self.ema_model.forward(x)
        if self.is_discrete:
            (indices,) = posteriors
        else:
            mu, logvar = posteriors

        psnr = self.psnr(reconstruction, x)
        ssim = self.ssim(reconstruction, x)

        ema_psnr = self.ema_psnr(ema_reconstruction, x)
        ema_ssim = self.ema_ssim(ema_reconstruction, x)

        if self.global_rank == 0 and batch_idx == 0:
            grid = make_grid(x.clamp(-1, 1))
            self.logger.log_image("val/val_images", [grid])

            grid = make_grid(reconstruction.clamp(-1, 1))
            self.logger.log_image("val/reconstruction_images", [grid])

        self.log_dict(
            {
                "val/psnr": psnr,
                "val/ssim": ssim,
                "val/ema_psnr": ema_psnr,
                "val/ema_ssim": ema_ssim,
            },
            prog_bar=True,
            sync_dist=True,
            on_epoch=True,
        )

    def configure_optimizers(self):
        # VAE optimizer settings
        optimizer_cfg = self.cfg.optimizer
        if optimizer_cfg.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=optimizer_cfg.lr,
                betas=optimizer_cfg.betas,
                weight_decay=optimizer_cfg.weight_decay,
            )
        elif optimizer_cfg.optimizer == "shampoo":
            optimizer = DistributedShampoo(
                self.model.parameters(),
                lr=optimizer_cfg.lr,
                betas=optimizer_cfg.betas,
                epsilon=1e-12,
                weight_decay=optimizer_cfg.weight_decay,
                precondition_frequency=optimizer_cfg.preconditioning_frequency,
                max_preconditioner_dim=optimizer_cfg.max_preconditioner_dim,
                start_preconditioning_step=optimizer_cfg.start_preconditioning_step,
                use_decoupled_weight_decay=True,
                grafting_config=AdamGraftingConfig(
                    beta2=optimizer_cfg.betas[-1],
                    epsilon=1e-8,
                ),
            )

        if optimizer_cfg.schedule == "cosine":
            vae_schedule = schedulers.create_cosine_with_warmup(
                optimizer,
                total_steps=optimizer_cfg.total_steps,
                warmup_steps=optimizer_cfg.warmup_steps,
                lr=optimizer_cfg.lr,
                warmup_lr=optimizer_cfg.warmup_lr,
                min_lr=optimizer_cfg.min_lr,
            )
        elif optimizer_cfg.schedule == "poly":
            vae_schedule = schedulers.create_poly_with_warmup(
                optimizer,
                total_steps=optimizer_cfg.total_steps,
                warmup_steps=optimizer_cfg.warmup_steps,
                lr=optimizer_cfg.lr,
                warmup_lr=optimizer_cfg.warmup_lr,
                power=optimizer_cfg.power,
            )

        if not self.use_adversarial_loss:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": vae_schedule,
                    "interval": "step",
                },
            }

        # Discriminator optimizer settings
        discriminator_cfg = optimizer_cfg.discriminator
        if discriminator_cfg.optimizer == "adamw":
            discriminator_optimizer = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=discriminator_cfg.lr,
                betas=discriminator_cfg.betas,
                weight_decay=discriminator_cfg.weight_decay,
            )

        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": vae_schedule,
                    "interval": "step",
                },
            },
            {
                "optimizer": discriminator_optimizer,
            },
        )


@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig) -> None:
    # Set seed
    L.seed_everything(cfg.seed)

    # checkpoint location
    save_loc = setup_checkpoint_dir(cfg)
    cfg.trainer.save_loc = save_loc

    # wandb logger
    if cfg.trainer.wandb:
        wandb_logger = WandbLogger(
            project=cfg.trainer.wandb_project, save_dir=cfg.trainer.wandb_save_dir
        )

    callbacks = []
    # lr monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # use rich printing
    if cfg.trainer.rich_print:
        callbacks.append(RichModelSummary())
        callbacks.append(RichProgressBar())

    trainer = L.Trainer(
        # distributed settings
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        precision=cfg.trainer.precision,
        strategy=cfg.trainer.strategy,
        # callbacks and logging
        callbacks=[lr_monitor, *callbacks],
        logger=wandb_logger if cfg.trainer.wandb else None,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        # training related
        overfit_batches=cfg.optimizer.overfit_batches,
        max_steps=cfg.optimizer.total_steps,
        use_distributed_sampler=cfg.dataset.use_distributed_sampler,
        benchmark=True,
        enable_checkpointing=False,
    )

    # create model
    vae_cfg = cfg.model.vae_type
    model_cfg = OmegaConf.to_container(cfg.model.config, resolve=True)
    if vae_cfg.type == "continuous":
        vae = VAE(prior=vae_cfg.latent, **model_cfg)
    elif vae_cfg.type == "discrete":
        vae = VQVAE(
            quantization=vae_cfg.quantization,
            levels=vae_cfg.levels,
            num_codebooks=vae_cfg.num_codebooks,
            embedding_dim=vae_cfg.embedding_dim,
            rotate=vae_cfg.rotate,
            **model_cfg,
        )

    if cfg.model.compile:
        vae.compile()

    # # create dataset
    train_transform = base_train_transform(
        cfg.dataset.image_size, cfg.dataset.augmentations.horizontal_flip
    )
    validation_transform = val_transform(
        cfg.dataset.image_size, cfg.dataset.max_crop_size
    )
    train_dataset, val_dataset, collate_fn = create_dataset(
        cfg.dataset.name,
        train_transform=train_transform,
        val_transform=validation_transform,
        num_proc=cfg.dataset.num_proc,
    )
    train_dataset = split_dataset_by_node(
        train_dataset, rank=trainer.global_rank, world_size=trainer.world_size
    )
    val_dataset = split_dataset_by_node(
        val_dataset, rank=trainer.global_rank, world_size=trainer.world_size
    )

    # create dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.dataset.val_batch_size,
        shuffle=False,
        num_workers=cfg.dataset.val_num_workers,
        collate_fn=collate_fn,
    )

    model = VAEModel(cfg, vae)

    # log config info to wandb
    if trainer.global_rank == 0:
        if cfg.trainer.wandb:
            wandb_logger.experiment.config.update(
                {
                    "seed": cfg.seed,
                    **flatten_dict(cfg.trainer),
                    **flatten_dict(cfg.model),
                    **flatten_dict(cfg.dataset),
                    **flatten_dict(cfg.loss),
                    **flatten_dict(cfg.optimizer),
                }
            )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
