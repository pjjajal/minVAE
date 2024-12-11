from torch.utils.data import default_collate


def image_collate_fn(batch, keys={"image": "jpg", "label": "cls"}):
    batch = default_collate(batch)
    return batch[keys["image"]], batch[keys["label"]]


def process_data(sample, transform=None, image_key="jpg"):
    if transform is not None:
        sample[image_key] = [transform(img.convert("RGB")) for img in sample[image_key]]
    return sample
