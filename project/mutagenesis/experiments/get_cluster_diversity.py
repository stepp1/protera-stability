from pathlib import Path
import multiprocessing
import contextlib

from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy.spatial.distance import hamming
from Bio import pairwise2, SeqIO

data_path = Path('../data/Protera')

def compute_diversity(seq1, seq2, method="hamming"):
    if method == "alignment":
        method = pairwise2.align.globalxx
        diversity = method(seq1, seq2)
        diversity = diversity[0].score / max(len(seq1), len(seq2))

    elif method == "hamming":
        if len(seq1) != len(seq2):
            size = min(len(seq1), len(seq2))
            seq1 = seq1[:size]
            seq2 = seq2[:size]
            
        method = hamming
        #print(list(seq1), "\n" ,list(seq2))
        diversity = method(list(seq1), list(seq2))
    
    return diversity


def div_vs_all(sequence, other_sequences, reducer = np.nanmean):
        v_diversity = np.vectorize(lambda x: compute_diversity(sequence, x))
        if len(other_sequences) > 1:
            div_vs_all = v_diversity(other_sequences)
        else:
            print(f"Skipping sequence {sequence}")
            return np.nan

        reduced_div_vs_all = reducer(div_vs_all) if len(div_vs_all) >= 1 else np.nan
        return reduced_div_vs_all
    
def dataset_diversity(sequences, method="hamming", reduce="mean", verbose=True):
    if reduce != "mean":
        raise NotImplementedError
    else:
        # nan results are due to different lengths
        reducer = np.nanmean

    reduced_divs = []
    if verbose:
        pbar = tqdm(total=len(sequences), miniters=1, smoothing=1)

    all_other_sequences = [
        np.concatenate((sequences[:idx], sequences[idx + 1 :]))
        for idx in range(len(sequences))
    ]  
    
    with contextlib.closing(multiprocessing.Pool(4)) as pool:
        for sequence_idx, sequence in enumerate(sequences):
            other_sequences = all_other_sequences[sequence_idx]
            reduced_divs.append(pool.apply_async(div_vs_all, args=(sequence, other_sequences)))
            
            
    for idx, result in enumerate(reduced_divs):
        reduced_divs[idx] = result.get()
        if verbose:
            pbar.update(1)
            pbar.refresh()
        
    pool.join()

    if verbose:
        pbar.close()

    return reduced_divs

def get_cluster_diversity():
    rep_seqs = dict()
    rep_seq_glob = list((data_path / "clustering").glob("cluster_*_rep_seq.fasta"))
    print(rep_seq_glob)

    for rep_seqs_pth in rep_seq_glob:
        df_name = rep_seqs_pth.stem.split("_")[1]

        df = pd.DataFrame()
        
        sequences = [str(record.seq) for record in SeqIO.parse(rep_seqs_pth, "fasta")]
        sequences = np.random.choice(sequences, 3000)
        # alignment = dataset_diversity(sequences, "alignment")
        hamming = dataset_diversity(sequences, "hamming")

        df["sequences"] = sequences
        # df["alignment"] = alignment
        df["hamming"] = hamming

        rep_seqs[df_name] = df

    return rep_seqs


if __name__ == "__main__":
    print("Running...")
    
    rep_seqs = get_cluster_diversity()

    for key, df in rep_seqs.items():
        df.to_csv(f"{key}.csv", index = False)


