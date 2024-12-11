from datasets import load_dataset
from functools import partial
from .utils import image_collate_fn, process_data

TOTAL_SAMPLES = 162_770
TOTAL_VAL_SAMPLES = 19_867
TOTAL_TEST_SAMPLES = 19_962
TOTAL_CLASSES = 10_177

URL = "flwrlabs/celeba"
KEYS = {"image": "image", "label": "celeb_id"}

def celeba_collate_fn(batch):
    return image_collate_fn(batch, keys=KEYS)

def celeba_train(transform=None, num_proc=1):
    dataset = load_dataset(URL, split="train", num_proc=num_proc)
    dataset = dataset.select_columns(list(KEYS.values()))
    dataset.set_transform(
        partial(process_data, transform=transform, image_key=KEYS["image"])
    )
    return dataset


def celeba_val(transform=None, num_proc=1):
    dataset = load_dataset(URL, split="valid", num_proc=num_proc)
    dataset = dataset.select_columns(list(KEYS.values()))
    dataset.set_transform(
        partial(process_data, transform=transform, image_key=KEYS["image"])
    )
    return dataset


def celeba_test(transform=None, num_proc=1):
    dataset = load_dataset(URL, split="test", num_proc=num_proc)
    dataset = dataset.select_columns(list(KEYS.values()))
    dataset.set_transform(
        partial(process_data, transform=transform, image_key=KEYS["image"])
    )
    return dataset
