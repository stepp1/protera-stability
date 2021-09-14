from protera_stability.config.lazy import LazyCall as L
import torch


def get_default_optimizer_params(model: torch.nn.Module):
    return model.parameters()


Adam = L(torch.optim.Adam)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
    ),
    lr=1e-3,
    weight_decay=0.01,
)

AdamW = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
    ),
    lr=1e-3,
    weight_decay=0.01,
)

SGD = L(torch.optim.SGD)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
    ),
    lr=2e-2,
    momentum=0.9,
    weight_decay=1e-4,
)
