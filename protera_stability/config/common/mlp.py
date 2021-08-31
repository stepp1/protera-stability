from protera_stability.config.lazy import LazyCall as L
from protera_stability.models import ProteinMLP
from torch import nn


model = L(ProteinMLP)(
    n_in=1280,
    n_units=1024,
    n_layers=3,
    act=L(nn.LeakyReLU)(negative_slope=0.01),
    drop_p=0.7,
    last_drop=False,
)
