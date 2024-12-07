import torch

class GaussianDistribution(torch.nn.Module):
    def __init__(self, min_logvar: float = -30.0, max_logvar: float = 20.0):
        super().__init__()
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

    def sample(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        return mean + torch.rand_like(mean) * std

    def forward(
        self, parameters
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        mean, logvar = torch.chunk(parameters, 2, dim=1)
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        return self.sample(mean, logvar), (mean, logvar)
