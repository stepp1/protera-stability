from pathlib import Path
import pickle as pkl
import random
import os

import torch
from torch.utils.data import DataLoader
import esm

from tqdm import tqdm
import pandas as pd
import numpy as np
import h5py


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class EmbeddingProtein1D:
    """
    Embedding Extractor for 1D protein sequences.
    """
    def __init__(self, model_name, open_func, data_path, gpu = False):
        """
        Constructor.
        
        Arguments:
        ---------
            model_name : str
                Can be one of the ESM models
                
            open_func : callable
                A callable function that loads the data into a train/test/val dictionary
                
            data_path: str or pathlib.Path
                The data's base path
                
            gpu : bool
                Whether to move the model to cuda.
                
        """
        
        self.model, self.alphabet = torch.hub.load("facebookresearch/esm", model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.open_func = open_func
        self.data_path = data_path
        self.data = None
        self.gpu = gpu
        self.kind = None
        if self.gpu:
            self.model.cuda()
            
    def open_embeddings(self, prefix, kind):
        """
        Opens a binary pkl file with the precomputed embeddings.
        """
        fname = (
            self.data_path
            / f'{prefix}_embeddings_{kind}.pkl'
        )
        return pkl.load(open(fname, 'rb'))
    
    
    def predict(self, sequence):
        """
        Obtains model's predictionsfor the given sequence.
        
        Arguments:
        ---------
            sequence: str
                An aminoacid sequence
                
        Returns:
        -------
            predictions : dict
        """
        # only one sequence
        if np.array(sequence).shape == ():
            sequence = sequence.upper()
            data = [("0", sequence)]
        else:
            data = sequence

        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        with torch.no_grad():
            if self.gpu:
                batch_tokens = batch_tokens.cuda()
            predictions = self.model(batch_tokens, repr_layers=[33])
        return predictions
        
        
    def get_embedding(self, sequence, sequence_emb = True):
        """
        Obtains the embeddings model predictions and averages over 
        the sequence length to  obtain embedding representations.
        
        Arguments:
        ---------
            sequence: str
                An aminoacid sequence
            sequence_emb: bool
                Whether to reduce the token embeddings to obtain a sequence embedding
                
        Returns:
        -------
            sequence_embeddings/token_embeddings: str/list
        """
        results = self.predict(sequence)
    
        token_embeddings = results["representations"][33] # shape: (bs, seq_len, emb_dim)
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.

        if sequence_emb: # one emb for the whole seq => average over seq_len
            if token_embeddings.shape[0] == 1:
                sequence_embeddings = (
                    token_embeddings[0, 1 : len(sequence) + 1].mean(dim=0).cpu().numpy()
                )
            else:
                sequence_embeddings = (
                    token_embeddings[:, 1 : len(sequence) + 1, :].mean(dim=1).cpu().numpy()
                )
            return sequence_embeddings
    
        return token_embeddings


    def generate_embeddings(self, file_prefix, save = False, kind = 'train', bs = 32, subset = None, data = None):
        """
        Generates sequence/token embeddings for a whole dataset.
        Can write the embeddings into a pickle file with the path_out argument.
        
        Arguments:
        ---------
            file_prefix : str, pathlib.Path
                A file prefix to load certain files from the data_path
                
            path_out : bool
                Whether to save the embeddings to a pickle file
                
            kind: str
                One of train/test/val sets
            
            bs: int
                The dataloader batch size
                
            subset: int
                A subset size of the train/test/val set
            
        """
        self.data = data
        self.data = self.data if self.data is None or self.kind != kind \
                              else self.open_func(self.data_path, file_prefix)[kind]
        self.kind = kind
    
        if subset is not None:
            random.choice(df.index)
            raise NotImplementedError

        # batch the dataset and return a batch of the form [(label, seq), ...]
        dl = DataLoader(
            list(self.data.itertuples(index=False, name=None)), 
            batch_size=bs, 
            collate_fn=lambda batch: batch
        )

        embeddings = {}
        for batch in tqdm(dl):
            batch_embeddings = self.get_embedding(batch)
            i = 0
            if len(batch_embeddings.shape) == 2:            
                for (label, seq), emb in zip(batch, batch_embeddings):
                    i +=1  
                    embeddings[seq] = emb
            else:
                label, seq = zip(*batch)
                embeddings[seq[0]] = batch_embeddings

        if save:
            out_fname = (
                self.data_path
                / f"{file_prefix}_embeddings_{kind}.pkl"
            )
            pkl.dump(embeddings, open(out_fname, 'wb'))
            print(f'Embeddings saved to {out_fname}')

        return embeddings
    
    
    def generate_datasets(self, file_prefix, save_emb = False, kind = 'train', bs = 32, subset = None, load_embeddings = False, overwrite = False, data = None):
        """
        Generates sequence/token embeddings for a whole dataset.
        Can write the embeddings into a pickle file with the path_out argument.
        
        Arguments:
        ---------
            file_prefix : str, pathlib.Path
                A file prefix to load certain files from the data_path
                
            save_emb : bool
                Save precomputed embeddings
                
            kind: str
                One of train/test/val sets
            
            bs: int
                The dataloader batch size
                
            subset: int
                A subset size of the train/test/val set
                
            load_embeddings : bool
                Whether to load the embeddings from a file in the data_path.
                It assumes that file a is binary pickle and the file name is f"{file_prefix}_embeddings.pkl"
        """
        if subset:
            raise NotImplementedError(f"Subsets are not implemented yet")
            
        self.data = data
        self.data = self.data if self.data is not None and self.kind == kind \
                              else self.open_func(self.data_path, file_prefix)[kind]
        
        if load_embeddings == False:
            embeddings = self.generate_embeddings(file_prefix, save=save_emb, kind=kind, bs=bs, subset=subset) 
            
        else:
            embeddings = self.open_embeddings(file_prefix, kind)
        
        self.kind = kind
        
        h5_fname = (
            self.data_path 
            / f"{file_prefix}_{kind}.h5"
        )
        
        if Path(h5_fname).exists() and overwrite:
            raise ValueError(f"Dataset {h5_fname} exists.")
            
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
                "sequences", (n_samples,), dtype= h5py.string_dtype(encoding='utf-8', length=None)
            )
            
            dset_labels.attrs["target"] = file_prefix
            dset_labels[:] = self.data.labels
            dset_embeddings[:] = list(embeddings.values())
            dset_seqs[:] = self.data.sequence.astype(str)
            
            
        return h5py.File(str(h5_fname), "r")