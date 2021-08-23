# Predicting Protein Stability with ML

## Problem
1. We want to let the lab know what proteins can have a higher stability in order to achieve a faster iterations.
2. In order to achieve this, we can build a ML-based regressor that predicts protein stability given the aminoacid sequence.
3. Nevertheles, labelled protein sequence data is scarce. More so, when we are interesed on stability annotations.

## Objective/Target
* With the two available datasets, [Rocklin et al. 2017](https://doi.org/10.1126/science.aan0693) and [Høie et al. 2021](https://doi.org/10.1101/2021.06.26.450037), we want to determine the minimum amount of input data that is needed to train ML-based protein stability regressors.

## Hypothesis
* If we use protein dataset with a higher degree of diversity, we can achieve better results with less data. Consequently, our control case will be based on random sampled subsets.

## Methodology 
* We will experiment with different subsets determined by two main sampling techniques:
	1. The first technique will to maximize diversity. Subset size can be determined by two factors: a minimum diversity score - e.g. our samples cannot have diversity score lower than $x$ - and maximum size given by a percent of the original dataset -e.g. our subset cannot have a size bigger than $|X| * $x$-.
	2. The second technique will create random sampled subsets that cannot have size bigger than a percent of the original dataset.

* Given the previous point, we need to do the achieve the following intermediate objectives:
	0. Perform EDA to gain a sense of what our data is about.
	1. Compute a diversity score for every sample.
	2. Extract pretrained embeddings as feature descriptors. 
	3. Create new dataset files where we can store our embeddings as 1 dimensional arrays.
	4. Write custom sampling techniques. 
	5. Create different experiment setups.

* As we have two datasets, we will also explore our model's transfer learning capabilities.

## EDA Results (to improve)
* Alignment Scores: Mathematically similar to Hamming Distance but more versatile.
* Score per different metric (length, uniqueness of aminoacids) 

## Preprocessing Pipeline

### Embeddings
1. Create batched dataloader for a given dataset. The batches must have the following form: `[(label, sequence), ...]`.
2. Retrieve token embeddings for a batch of sequences.
3. Reduce the token embeddings to sequence embeddings (average over sequence dimension).
4. We have embeddings per sequence.

### Diversity
#### TODO

## Results 
#### TODO

### Using
### Using
### Using

## Papers used

|    Shorthand   |             Paper           |      Dataset     |     Description      |    Use    |
|----------------|-----------------------------|------------------|----------------------|-----------|
| parallel       | [Rocklin et al. 2017](https://doi.org/10.1126/science.aan0693)  |    -  | 1D Protein Sequences with the custom stability scores(?).                 | Input Data to ESM1-b |
| mutagenesis    | [Høie et al. 2021](https://doi.org/10.1101/2021.06.26.450037)   |    -  | 1D Protein Sequences with their ddG values annotated by Rosseta.          | Input Data to ESM1-b + Transfer Learning |
| ESM-1b         | [Rives et al. 2019](https://doi.org/10.1101/622803)             | UR50  | Pretrained SOTA general-purpose protein language model. Can be used to predict structure, function and other protein properties directly from individual sequences. | We pass protein sequences as input and extract embeddings that are used as feature descriptors. We attempt to train one or more models that can predict protein stability. |
| MSA Transformer| [Rao et al. 2021]https://doi.org/10.1101/2021.02.12.430858 )    |    -  | TODO                                                                      | They report improved results when maximizing diversity as Hamming Distance. |

