import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal

def reconstruction_loss(type: Literal['l2', 'l1', 'l1-smooth']):
    match type:
        case 'l2':
            return nn.MSELoss()
        case 'l1':
            return nn.L1Loss()
        case 'l1-smooth':
            return nn.SmoothL1Loss()
        case _:
            raise ValueError(f'Unknown loss type: {type}')
    
def reg_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def discriminator_loss(real, fake, type: Literal['bce', 'hinge']):
    if type == "bce":
        loss_real = F.binary_cross_entropy_with_logits(real, torch.ones_like(real))
        loss_fake = F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
    elif type == "hinge":
        loss_real = F.relu(1.0 - real).mean()
        loss_fake = F.relu(1.0 + fake).mean()
    else:
        raise ValueError(f'Unknown loss type: {type}')
    return 0.5 * (loss_real + loss_fake) 

def generator_loss(fake, type: Literal['bce', 'hinge']):
    if type == "bce":
        return F.binary_cross_entropy_with_logits(fake, torch.ones_like(fake))
    elif type == "hinge":
        return -fake.mean()
    else:
        raise ValueError(f'Unknown loss type: {type}')