import torch.optim as optim


def create_linear_warmup(optimizer, lr, warmup_lr, warmup_steps):
    return optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_lr / lr, end_factor=1, total_iters=warmup_steps
    )


def create_cosine_with_warmup(
    optimizer, total_steps, warmup_steps, lr, warmup_lr, min_lr=0.0
):
    warmup_schedule = create_linear_warmup(optimizer, lr, warmup_lr, warmup_steps)
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_schedule, cosine_schedule], milestones=[warmup_steps]
    )


def create_poly_with_warmup(
    optimizer, total_steps, warmup_steps, lr, warmup_lr, power=0.9
):
    warmup_schedule = create_linear_warmup(optimizer, lr, warmup_lr, warmup_steps)
    poly_schedule = optim.lr_scheduler.PolynomialLR(
        optimizer, total_steps - warmup_steps, power=power
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_schedule, poly_schedule], milestones=[warmup_steps]
    )
