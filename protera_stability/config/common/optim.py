from protera_stability.config.lazy import LazyCall as L
import torch


def get_default_optimizer_params(model: torch.nn.Module):
    return model.parameters()


Adam = L(torch.optim.Adam)(
    lr=1e-3,
    weight_decay=1e-2,
)

AdamW = L(torch.optim.AdamW)(
    lr=1e-3,
    weight_decay=1e-2,
)

SGD = L(torch.optim.SGD)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
    ),
    lr=0.02,
    momentum=0.9,
    weight_decay=1e-4,
)
