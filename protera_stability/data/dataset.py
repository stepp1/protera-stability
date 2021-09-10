import h5py
import torch
from sklearn import preprocessing

from protera_stability.proteins.embeddings import EmbeddingExtractor1D
from protera_stability.utils import dim_reduction


class ProteinStabilityDataset(torch.utils.data.Dataset):
    """Protein1D Stability Dataset."""

    def __init__(self, proteins_path, ret_dict=False):
        """
        Args:
            proteins_path (string): Path to the H5Py file that contains sequences and embeddings.
            ret_dict (bool): If True, it will return a dictionary as batch. Otherwise, X and y tensors will be returned.
        """
        self.stability_path = proteins_path
        self.ret_dict = ret_dict
        self.x_scaler = preprocessing.StandardScaler()
        self.y_scaler = preprocessing.StandardScaler()

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
            "embeddings": torch.from_numpy(embeddings),
            "labels": torch.Tensor([labels]),
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
