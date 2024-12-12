from datasets import load_dataset
from functools import partial
from .utils import image_collate_fn, process_data

TOTAL_SAMPLES = 28_000
TOTAL_VAL_SAMPLES = 28_000
TOTAL_CLASSES = 19

URL = "korexyz/celeba-hq-256x256"
KEYS = {"image": "image", "label": "label"}

def celeba_hq_collate_fn(batch):
    return image_collate_fn(batch, keys=KEYS)

def celeba_hq_train(transform=None, num_proc=1):
    dataset = load_dataset(URL, split="train", num_proc=num_proc)
    dataset = dataset.select_columns(list(KEYS.values()))
    dataset.set_transform(
        partial(process_data, transform=transform, image_key=KEYS["image"])
    )
    return dataset


def celeba_hq_val(transform=None, num_proc=1):
    dataset = load_dataset(URL, split="validation", num_proc=num_proc)
    dataset = dataset.select_columns(list(KEYS.values()))
    dataset.set_transform(
        partial(process_data, transform=transform, image_key=KEYS["image"])
    )
    return dataset