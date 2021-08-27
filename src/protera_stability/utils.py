import copy

import matplotlib.pyplot as plt
import pandas as pd
import torch
from IPython.display import clear_output, display
from pytorch_lightning import Callback
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .embeddings import EmbeddingExtractor1D


def unwrap(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    return x


class PrintCallback(Callback):
    def __init__(self):
        self.metrics = []
        self.cur_epoch = 0

    def on_epoch_end(self, trainer, pl_module):
        clear_output(wait=True)
        metrics_dict = copy.deepcopy(trainer.callback_metrics)
        
        #del metrics_dict["loss"]
        metrics_dict = {k: unwrap(v) for k, v in metrics_dict.items()}
        metrics_dict["epoch"] = self.cur_epoch
        self.cur_epoch += 1
        self.metrics.append(metrics_dict)
        
        del metrics_dict
        
        
        # column-names should be modified as per your usage
        metrics_df = pd.DataFrame.from_records(
            self.metrics,
            columns=[
                "epoch",
                "train_loss",
                "valid_loss",
                "valid_r2",
            ],
        )
        display(metrics_df)

def dim_reduction(
    X,
    y=None,
    strategy="PCA",
    n_components=2,
    prefix=None,
    plot_viz=True,
    save_viz=False,
):
    valid_strats = ("PCA", "UMAP", "TSNE")
    if strategy not in valid_strats:
        raise ValueError(f"{strategy} is not a valid dimensionality reduction strategy")

    if strategy == valid_strats[0]:
        reducer = PCA(n_components=n_components)
    elif strategy == valid_strats[1]:
        raise NotImplementedError
        # reducer = sklearn.decomposition.UMAP(n_components=2)
    elif strategy == valid_strats[2]:
        reducer = TSNE(n_components=2)

    X_hat = reducer.fit_transform(X)

    if plot_viz:
        f, ax = plt.subplots(figsize=(10, 5))
        scatter = ax.scatter(X_hat[:, 0], X_hat[:, 1], c=y)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

        cb = plt.colorbar(scatter, spacing="proportional")
        cb.set_label(prefix)
        plt.show()

    if save_viz:
        fname = f"{strategy}.png"
        print(f"Saved as {fname}")
        plt.savefig(fname, dpi=300)

    return X_hat

def load_dataset_raw(
    data_path, kind=None, reduce=False, scale=True, to_torch=False, close_h5=True, verbose=False
):
    args_dict = {
        "model_name": "esm1b_t33_650M_UR50S",
        "base_path": data_path,
        "gpu": False,
    }

    emb_stabilty = EmbeddingExtractor1D(**args_dict)
    task = "stability"

    if kind is not None:
        task = f"{task}_{kind}"

    if verbose:
        print(f"Using: {data_path / task}.csv, {data_path / task}.h5, {data_path / task}_embeddings.pkl")

    dset = emb_stabilty.generate_datasets([f"{task}.csv"], h5_stem=task, embedding_file=f"{task}_embeddings")

    X, y = dset["embeddings"][:].astype("float32"), dset["labels"][:].astype("float32")

    if reduce:
        X = dim_reduction(X, y, plot_viz=False)

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        scaler = StandardScaler()
        y = scaler.fit_transform(y.reshape(-1, 1)).reshape(y.shape)

    if to_torch:
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)

    if close_h5:
        dset.close()
        return X, y
    return X, y, dset
