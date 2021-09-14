import pathlib
import pickle as pkl
import sys
from pathlib import Path
from typing import Dict, List, Union

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class EmbeddingExtractor1D:
    """
    Embedding Extractor for 1D protein sequences.
    """

    def __init__(self, model_name, base_path, gpu=False):
        """
        Constructor.

        Arguments:
        ---------
            model_name : str
                Must be one of the ESM pretrained models.

            base_path : str or pathlib.Path
                The data's base path.

            gpu : bool
                Whether to move the model to cuda.
        """

        self.model, self.alphabet = torch.hub.load("facebookresearch/esm", model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.base_path = base_path if isinstance(base_path, Path) else Path(base_path)
        self.data = None
        self.device = gpu

        if self.device:
            self.device = "cuda" if isinstance(self.device, bool) else self.device
        else:
            self.device = "cpu"
        self.model.to(self.device)

    def open_embeddings(self, filename: str) -> Dict[str, np.ndarray]:
        """
        Opens a binary pkl file with the precomputed embeddings.
        """
        fname = self.base_path / f"{filename}.pkl"
        return pkl.load(open(fname, "rb"))

    def predict(self, sequence):
        """
        Obtains model's predictionsfor the given sequence.

        Arguments:
        ---------
            sequence: str
                An aminoacid sequence.

        Returns:
        -------
            predictions : dict.
        """
        # only one sequence
        if np.array(sequence).shape == ():
            sequence = sequence.upper()
            data = [("0", sequence)]
        else:
            data = sequence
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        with torch.no_grad():
            batch_tokens = batch_tokens.to(self.device)
            predictions = self.model(batch_tokens, repr_layers=[33])
        return predictions

    def get_embedding(self, sequence, sequence_emb=True):
        """
        Obtains the embeddings model predictions and averages over
        the sequence length to  obtain embedding representations.

        Arguments:
        ---------
            sequence: str
                An aminoacid sequence.
            sequence_emb: bool
                Whether to reduce the token embeddings to obtain a sequence embedding.

        Returns:
        -------
            sequence_embeddings/token_embeddings: str/list.
        """
        results = self.predict(sequence)

        token_embeddings = results["representations"][
            33
        ]  # shape: (bs, seq_len, emb_dim)
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.

        if sequence_emb:  # one emb for the whole seq => average over seq_len
            if token_embeddings.shape[0] == 1:
                sequence_embeddings = (
                    token_embeddings[0, 1 : len(sequence) + 1].mean(dim=0).cpu().numpy()
                )
            else:
                sequence_embeddings = (
                    token_embeddings[:, 1 : len(sequence) + 1, :]
                    .mean(dim=1)
                    .cpu()
                    .numpy()
                )
            return sequence_embeddings

        return token_embeddings

    def generate_embeddings(
        self,
        files: List[Union[str, pathlib.Path]],
        path_out: Union[str, pathlib.Path] = None,
        bs: int = 32,
        subset: float = None,
        data: pd.DataFrame = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generates sequence/token embeddings for a whole dataset.
        Can write the embeddings into a pickle file with the path_out argument.

        Arguments:
        ---------
            files : List[Union[str, pathlib.Path]]
                A list of files .csv from the base_path to load.
                The files must contain two columns:  `labels` and `sequences`.

            path_out : Union[str, pathlib.Path]
                Filename to pickle the embeddings into.

            bs: int
                The dataloader batch size.

            subset: float
                A percentage of the original data to use as subset.

            data: pd.DataFrame
                A DataFrame to use instead of loading data.
                The files must contain two columns: `labels` and `sequences`.

        """
        self.data = data
        self.data = (
            self.data
            if self.data is not None
            else pd.concat([pd.read_csv(self.base_path / file) for file in files])
        )

        if subset is not None:
            # random.choice(df.index)
            raise NotImplementedError

        # batch the dataset and return a batch of the form [(label, seq), ...]
        dl = torch.utils.data.DataLoader(
            list(self.data.itertuples(index=False, name=None)),
            batch_size=bs,
            collate_fn=lambda batch: batch,
        )

        embeddings = {}
        for batch in tqdm(dl):
            batch_embeddings = self.get_embedding(batch)
            i = 0
            if len(batch_embeddings.shape) == 2:
                for (label, seq), emb in zip(batch, batch_embeddings):
                    i += 1
                    embeddings[seq] = emb
            else:
                label, seq = zip(*batch)
                embeddings[seq[0]] = batch_embeddings

        if path_out is not None:
            out_fname = self.base_path / f"{path_out}.pkl"
            pkl.dump(embeddings, open(out_fname, "wb"))
            print(f"Embeddings saved to {out_fname}")

        return embeddings

    def generate_datasets(
        self,
        files: List[Union[str, pathlib.Path]],
        h5_stem: str,
        embedding_to_save: List[Union[str, pathlib.Path]] = None,
        bs: int = 32,
        subset: float = None,
        embedding_file: Union[str, pathlib.Path] = None,
        overwrite: bool = False,
        data=None,
        target_name: str = "",
    ) -> h5py.Dataset:
        """
        Generates sequence/token embeddings for a whole dataset.
        Can write the embeddings into a pickle file with the path_out argument.

        Arguments:
        ---------
            files : List[Union[str, pathlib.Path]]
                A list of files .csv from the base_path to load.
                The files must contain two columns:  `labels` and `sequences`.

            h5_stem : str
                The stem to use for the generated dataset.

            embedding_to_save : Union[str, pathlib.Path]
                Filename to save precomputed embeddings.

            bs: int
                The dataloader batch size.

            subset: float
                A percentage of the original data to use as subset.

            embedding_file :  Union[str, pathlib.Path]
                A pickle file to load the embeddings from.
                It assumes the file is a pickle binary.
                Must be of same length than the .csv files

            overwrite: bool
                Whether to overwrite an existing h5 dataset.

            data: pd.DataFrame
                A DataFrame to use instead of loading data.
                The files must contain two columns: `labels` and `sequences`.

            target_name: str
                What the label column represents.
                This will be saved as an attrs of the h5.
        """
        if subset:
            raise NotImplementedError(f"Subsets are not implemented yet")

        self.data = data
        self.data = (
            self.data
            if self.data is not None
            else pd.concat([pd.read_csv(self.base_path / file) for file in files])
        )

        h5_fname = self.base_path / f"{h5_stem}.h5"
        if Path(h5_fname).exists() and not overwrite:
            print("Returning existing dataset...", file=sys.stderr)
            return h5py.File(str(h5_fname), "r")

        if embedding_file is None:
            embeddings = self.generate_embeddings(
                files,
                path_out=self.base_path / embedding_to_save if embedding_to_save is not None else None,
                bs=bs,
                subset=subset,
                data=data,
            )

        else:
            embeddings = self.open_embeddings(embedding_file)

        n_samples = len(embeddings)

        emb_len = max([len(val) for val in embeddings.values()])

        with h5py.File(str(h5_fname), "w") as file_handler:
            dset_labels = file_handler.create_dataset(
                "labels", (n_samples,), dtype=self.data.labels.dtype
            )
            dset_embeddings = file_handler.create_dataset(
                "embeddings", (n_samples, emb_len)
            )
            dset_seqs = file_handler.create_dataset(
                "sequences",
                (n_samples,),
                dtype=h5py.string_dtype(encoding="utf-8", length=None),
            )

            dset_labels.attrs["target"] = target_name
            dset_labels[:] = self.data.labels
            dset_embeddings[:] = list(embeddings.values())
            dset_seqs[:] = self.data.sequences.astype(str)

        return h5py.File(str(h5_fname), "r")
