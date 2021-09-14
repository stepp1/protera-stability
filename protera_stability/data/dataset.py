import dill
import pickle

from time import time
from pathlib import Path
import h5py
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing

from protera_stability.proteins.embeddings import EmbeddingExtractor1D
from protera_stability.utils import dim_reduction


class ProteinStabilityDataset(torch.utils.data.Dataset):
    """Protein1D Stability Dataset."""

    def __init__(self, proteins_path, otf_getter=None, ret_dict=False, otf_getter_args=(False,)):
        """
        Args:
            proteins_path (string): Path to the H5Py file that contains sequences and embeddings.
            on_the_fly_getter (cls): If not None, it will instantiate an object that computes an X's item on the fly. 
            ret_dict (bool): If True, it will return a dictionary as batch. Otherwise, X and y tensors will be returned.
        """
        self.stability_path = proteins_path
        self.ret_dict = ret_dict
        self.x_scaler = preprocessing.StandardScaler()
        self.y_scaler = preprocessing.StandardScaler()

        if otf_getter is not None:
            self.otf_getter = otf_getter.initialize(*otf_getter_args)
            self.sequences = self.otf_getter.sequences
            self.X = self.otf_getter.X()
            self.y = self.otf_getter.y()

        else:
            with h5py.File(str(self.stability_path), "r") as dset:
                self.sequences = dset["sequences"][:]
                X = dset["embeddings"][:].astype("float32")
                y = dset["labels"][:].astype("float32")

                self.X = self.x_scaler.fit_transform(X)
                self.y = self.y_scaler.fit_transform(y.reshape(-1, 1)).reshape(y.shape)

        self.indices = list(range(len(self.X)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sequences = self.sequences[idx]
        embeddings = self.X[idx]
        labels = self.y[idx]

        sample = {
            "sequences": sequences,
            "embeddings": torch.from_numpy(embeddings.astype("float32")),
            "labels": torch.from_numpy(np.array([labels]).astype("float32")),
        }

        return sample if self.ret_dict else (sample["embeddings"], sample["labels"])


def load_dataset_raw(
    data_path,
    kind=None,
    reduce=False,
    scale=True,
    to_torch=False,
    close_h5=True,
    verbose=False,
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
        print(
            f"Using: {data_path / task}.csv, {data_path / task}.h5, {data_path / task}_embeddings.pkl"
        )

    dset = emb_stabilty.generate_datasets(
        [f"{task}.csv"], h5_stem=task, embedding_file=f"{task}_embeddings"
    )

    X, y = dset["embeddings"][:].astype("float32"), dset["labels"][:].astype("float32")

    if reduce:
        X = dim_reduction(X, y, plot_viz=False)

    if scale:
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)

        scaler = preprocessing.StandardScaler()
        y = scaler.fit_transform(y.reshape(-1, 1)).reshape(y.shape)

    if to_torch:
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)

    if close_h5:
        dset.close()
        return X, y
    return X, y, dset

class AsSubscriptable:
    def __init__(self, values, get_func):
        self.values = values
        self.get_func = get_func

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return self.get_func(self.values, idx)


class EmbeddingGetter:
    def __init__(self, protein_csv_path, cuda, data_path):
        self.extractor_kwargs = {
            "model_name": "esm1b_t33_650M_UR50S",
            "base_path": data_path,
            "gpu": cuda,
        }
        self.extractor = EmbeddingExtractor1D(**self.extractor_kwargs)
        self.df = self.process_csv(protein_csv_path)
        self.sequences = self.df.sequences.values
        self.labels = self.df.labels.values

        self.ready = False
        self.X_values = None
        self.y_values = None

    def initialize(self, precompute=False):
        if self.ready:
            return self
        self.x_scaler = preprocessing.StandardScaler()
        self.y_scaler = preprocessing.StandardScaler()

        if precompute:
            print("Precomputing embeddings...")
            tmp_dir = Path("tmp") / str(int(time()))
            if not (self.extractor_kwargs["base_path"] / tmp_dir).exists():
                (self.extractor_kwargs["base_path"] / tmp_dir).mkdir(parents=True, exist_ok=True)

            dset = self.extractor.generate_datasets(
                [],
                data=self.df,   
                h5_stem=tmp_dir / "dataset",  # data_path / tmp_dir / dataset.h5
                bs=32,
                target_name="stability_scores"
            )

            embeddings = dset["embeddings"][:].astype("float32")
            y = dset["labels"][:].astype("float32")
            self.X_values = self.x_scaler.fit_transform(embeddings)
            self.y_values = self.y_scaler.fit_transform(y.reshape(-1, 1)).reshape(y.shape)

            dset.close()
            self.extractor.model.to("cpu")
            del self.extractor

        else:
            print("Initializing on the fly scalers...")
            # use 10% of the data to fit scaler
            seqs = np.random.choice(self.sequences, size=min(int(len(self.sequences) * 0.1), 500), replace=False)
            embeddings = []
            for seq in tqdm(seqs):
                embeddings.append(self.extractor.get_embedding(seq))

            y = self.labels

            self.x_scaler.fit(embeddings)
            self.y_scaler.fit(y.reshape(-1, 1))

        self.ready = True

        return self

    def __len__(self):
        return len(self.sequences)

    def process_csv(self, path):
        df = pd.read_csv(path)
        df = df.drop_duplicates().dropna()
        df = df[["variant", "Rosetta_ddg_score_02"]]
        df.columns = ["sequences", "labels"]
        df = df[df.columns[::-1]]

        return df

    def get_func_x(self, x, idx):
        if self.X_values is None:
            embedding = self.extractor.get_embedding(x[idx]).reshape(1, -1)
            out = self.x_scaler.transform(embedding).reshape(1280,)
        else:
            out = self.X_values[idx]
        return out

    def get_func_y(self, x, idx):
        if self.y_values is None:
            out = self.y_scaler.transform(x[idx].reshape(1, -1)).reshape(x[idx].shape)
        else:
            out = self.y_values[idx]
        return out

    def X(self):
        if not self.ready:
            raise Exception("You need to initialize the object. Run .initialize()")
        # lambda x, idx: self.x_scaler.transform(self.extractor.get_embedding(x[idx]).reshape(1, -1)).reshape(x.shape)
        return AsSubscriptable(self.sequences, self.get_func_x)
    
    def y(self):
        if not self.ready:
            raise Exception("You need to initialize the object. Run .initialize()")
        # lambda x, idx: self.y_scaler.transform(x[idx].reshape(1, -1).reshape(x.shape))
        return AsSubscriptable(self.labels, self.get_func_y)