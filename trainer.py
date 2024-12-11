import random
from datetime import datetime
from functools import partial
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
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
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

import utils.loss as losses
import utils.schedulers as schedulers
from dataset.celeba import celeba_collate_fn, celeba_train, celeba_val
from dataset.transforms import base_train_transform, val_transform
from dataset.utils import image_collate_fn
from models.gan import DinoPatchDiscriminator
from models.vae import VAE

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

        # manual optimization.
        self.automatic_optimization = False

        # which losses to use
        self.use_adversarial_loss = cfg.loss.adversarial_loss.enable
        self.use_perceptual_loss = cfg.loss.perceptual_loss.enable

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
            self.discriminator = DinoPatchDiscriminator(
                cfg.loss.adversarial_loss.discriminator
            )

        # Perceptual loss
        if self.use_perceptual_loss:
            if cfg.loss.perceptual_loss.type == "lpips":
                self.perceptual_loss = LearnedPerceptualImagePatchSimilarity(
                    net_type=cfg.loss.perceptual_loss.model_name,
                )
            elif cfg.loss.perceptual_loss.type == "dreamsim":
                self.perceptual_loss = dreamsim(
                    dreamsim_type=cfg.loss.perceptual_loss.model_name,
                    pretrained=True,
                    device=self.device,
                )

        # metrics
        self.psnr = PeakSignalNoiseRatio()
        self.ssim = StructuralSimilarityIndexMeasure()

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        if self.use_adversarial_loss:
            vae_opt, discriminator_opt = optimizers
        else:
            vae_opt = optimizers
        vae_schedule = self.lr_schedulers()

        x, label = batch

        # vae forward pass
        reconstruction, (mu, logvar) = self.model.forward(x)

        # optimize discriminator
        d_loss = 0.0
        if self.use_adversarial_loss:
            d_pred_real = self.discriminator(x)
            d_pred = self.discriminator(reconstruction.detach())
            d_loss = self.adverasarial_loss(d_pred_real, d_pred)
            discriminator_opt.zero_grad()
            self.manual_backward(d_loss)
            discriminator_opt.step()

        # optimize vae
        if self.cfg.optimizer.discriminator.discriminator_warmup < self.global_step or not self.use_adversarial_loss:
            reconstruction_loss = self.reconstruction_loss(reconstruction, x)
            kl_loss = self.kl_weight * self.kl_loss(mu, logvar)
            vae_loss = reconstruction_loss + kl_loss

            g_loss = 0.0
            if self.use_adversarial_loss:
                d_pred = self.discriminator(reconstruction)
                g_loss = self.adversarial_weight * self.generator_loss(d_pred)
                vae_loss += g_loss
            perceptual_loss = 0.0
            if self.use_perceptual_loss:
                perceptual_loss = self.perceptual_weight * self.perceptual_loss(
                    reconstruction.clamp(-1, 1), x
                )
                vae_loss += perceptual_loss

            vae_opt.zero_grad()
            self.manual_backward(vae_loss)
            vae_opt.step()
            vae_schedule.step()
        else:
            reconstruction_loss = 0.0
            kl_loss = 0.0
            vae_loss = 0.0
            perceptual_loss = 0.0
            g_loss = 0.0

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
        reconstruction, (mu, logvar) = self.model.forward(x)

        psnr = self.psnr(reconstruction, x)
        ssim = self.ssim(reconstruction, x)

        grid = make_grid(x.clamp(-1, 1))
        self.logger.log_image("val/val_images", [grid])

        grid = make_grid(reconstruction.clamp(-1, 1))
        self.logger.log_image("val/reconstruction_images", [grid])

        self.log_dict(
            {
                "val/psnr": psnr,
                "val/ssim": ssim,
            },
            prog_bar=True,
            sync_dist=True,
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
                self.discriminator.trainable_parameters(
                    head_only=discriminator_cfg.head_only
                ),
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
        wandb_logger = WandbLogger(project=cfg.trainer.wandb_project)

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
    )

    # create model
    model_cfg = OmegaConf.to_container(cfg.model.config, resolve=True)
    vae = VAE(**model_cfg)

    if cfg.model.compile:
        vae.compile()

    # create dataset
    train_dataset = celeba_train(
        transform=base_train_transform(
            cfg.dataset.image_size, cfg.dataset.augmentations.horizontal_flip
        ),
    )
    val_dataset = celeba_val(
        transform=val_transform(
            cfg.dataset.image_size,
            cfg.dataset.max_crop_size,
        ),
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
        collate_fn=celeba_collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        collate_fn=celeba_collate_fn,
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
