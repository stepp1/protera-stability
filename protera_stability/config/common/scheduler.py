from protera_stability.config.lazy import LazyCall as L
import torch

StepLR = L(torch.optim.lr_scheduler.StepLR)(
    step_size=5,
    gamma=1e-5,
)

CosineLR = L(torch.optim.lr_scheduler.CosineAnnealingLR)(
    T_max=70,
)
