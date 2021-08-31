import copy
from omegaconf import OmegaConf
import torch

from protera_stability.config.lazy import LazyCall as L
from protera_stability.config.instantiate import instantiate
from protera_stability.config.common.train import base_train
from protera_stability.config.common.data import base_dataset, base_dataloader, base_sampler, get_train_val_indices
from protera_stability.models import ProteinMLP

def default_setup():

    conf = base_train
    return conf

# TODO: SHOULD THIS BE IN configs/....py?
def setup_diversity(diversity_cutoff: float, random_percent: float, sequence_sampling: str = ""):
    """
    Basic Setup for Protein Diversity Experiments.

    Args:
        sequence_sampling (str): One of ["", "diversity", "random"]
        diversity_cutoff (float):
        random_percent (float):
    """
    conf = default_setup()
    conf.sequence_sampling = sequence_sampling
    conf.experiment = OmegaConf.create()
    conf.experiment.diversity_cutoff = diversity_cutoff
    conf.experiment.random_percent = random_percent
    conf.experiment.model = L(ProteinMLP)(
        n_in: int,
        n_units: int,
        n_layers: 3,
        act = L(torch.nn.LeakyRelu.),
        drop_p = 0.7,
        last_drop = False
    )



def setup_data(conf, base_dataset = base_dataset, base_sampler = base_sampler, base_dl = base_dataloader):
    dataset = instantiate(base_dataset.data)
    train_idx, valid_idx = get_train_val_indices(
        dataset,
        conf.random_split
    )

    # Check training sampling
    train_sampler = copy(base_sampler)
    if conf.sequence_sampling == "":
        # this is because we aren't using a "special" sampling method, therefore we directly pass the indices
        train_sampler.random.indices = train_idx
        # sampler = instantiate(sampler.random)

    elif conf.sequence_sampling == "diversity":
        train_sampler.diversity.set_sequences.dataset = dataset
        train_sampler.diversity.set_sequences.set_indices = train_idx
        train_sampler.diversity.max_size = int(len(dataset) * 0.8)
        # sampler = instantiate(sampler.diversity)

    elif conf.sequence_sampling == "random":
        train_sampler.random.indices.dataset = dataset
        train_sampler.random.indices.set_indices = train_idx
        # sampler = instantiate(sampler.random)

    else:
        raise ValueError(f"Sequence Sampling Method {conf.sequence_sampling} is not valid.")
    
    # just pass the indices to the sampler
    valid_sampler = copy(base_sampler)
    valid_sampler.random.indices = valid_idx

    # dataloaders
    base_dl.train.sampler = train_sampler
    base_dl.valid.sampler = valid_sampler

    conf.dataloader = base_dl
    return conf


def setup_trainer():
    ...

