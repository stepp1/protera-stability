# Predicting Protein Stability with ML

## EDA Results
* Alignment Scores: Same as Hamming Distance but more versatile. Unfinished.
* Score per different metric (length, uniqueness of aminoacids) 

## Preprocessing Pipeline
1. Create batched dataloader for a given dataset. The batches must have the following form: `[(label, sequence), ...]`.
2. Retrieve token embeddings for a batch of sequences.
3. Reduce the token embeddings to sequence embeddings (average over sequence dimension).
4. We have embeddings per sequence.


## Papers used

| Shorthand | Paper           | Dataset | Use  |
|-----------|-----------------------------|---------|--------------|
| ESM-1b    | [Rives et al. 2019](https://doi.org/10.1101/622803)  | UR50  | Pretrained Embedding.SOTA general-purpose protein language model. Can be used to predict structure, function and other protein properties directly from individual sequences. |
