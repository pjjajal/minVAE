import torch
import torchvision.transforms.v2 as tvt


def base_train_transform(image_size: int, hflip: bool = True):
    return tvt.Compose(
        [
            tvt.ToImage(),
            tvt.ToDtype(torch.uint8, scale=True),
            tvt.Resize(image_size, antialias=True),
            tvt.RandomCrop(image_size),
            tvt.RandomHorizontalFlip() if hflip else tvt.Identity(),
            tvt.ToDtype(torch.float32, scale=True),
            tvt.Normalize(mean=[0.5] * 3, std=[0.5] * 3), # [-1, 1] normalization
        ]
    )


def val_transform(image_size: int, max_crop_size: int):
    return tvt.Compose(
        [
            tvt.ToImage(),
            tvt.ToDtype(torch.uint8, scale=True),
            tvt.Resize(image_size, antialias=True),
            tvt.CenterCrop(max_crop_size),
            tvt.ToDtype(torch.float32, scale=True),
            tvt.Normalize(mean=[0.5] * 3, std=[0.5] * 3), # [-1, 1] normalization
        ]
    )