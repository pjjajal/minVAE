from .celeba_hq import celeba_hq_train, celeba_hq_val, celeba_hq_collate_fn
from .celeba import celeba_train, celeba_val, celeba_collate_fn
from .imagenet import imagenet_train, imagenet_val, imagenet_collate_fn


def create_dataset(dataset, train_transform=None, val_transform=None, num_proc=1):
    train_dataset = None
    val_dataset = None
    collate_fn = None
    if dataset == "celeba":
        train_dataset = celeba_train(transform=train_transform, num_proc=num_proc)
        val_dataset = celeba_val(transform=val_transform, num_proc=num_proc)
        collate_fn = celeba_collate_fn
    elif dataset == "celeba-hq":
        train_dataset = celeba_hq_train(transform=train_transform, num_proc=num_proc)
        val_dataset = celeba_hq_val(transform=val_transform, num_proc=num_proc)
        collate_fn = celeba_hq_collate_fn
    elif dataset == "imagenet":
        train_dataset = imagenet_train(transform=train_transform, num_proc=num_proc)
        val_dataset = imagenet_val(transform=val_transform, num_proc=num_proc)
        collate_fn = imagenet_collate_fn
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return train_dataset, val_dataset, collate_fn