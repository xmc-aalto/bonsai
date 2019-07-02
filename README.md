# Bonsai

C++ implementation of the paper [Bonsai - Diverse and Shallow Trees for Extreme Multi-label Classification](https://arxiv.org/abs/1904.08249)

# Introduction

## Bonsai - Diverse and Shallow Trees for Extreme Multi-label Classification

- Extreme Multi-Label Classification (**XMC**) refers to supervised learning of a classifier which can automatically label an instance with a subset of relevant labels from an extremely large set of all possible target labels. *Bonsai* learns an ensemble of diversity promoting shallow trees which achieves accuracies at par with the state-of-the-art one-vs-rest methods while being much faster to train.
- A tree built by *Bonsai* differs from one built by *Parabel* in the following aspects,
  1. A larger fanout at every node i.e instead of the two-way partioning of labels done by *Parabel*, *Bonsai* does a k-way partitioning. (k was set to 100 for most of our experiements)
  2. Allowing unbalanced partitioning of labels at every internal tree node thus producing a more data-dependent partitioning.
  3. Shallower i.e having much lesser number of levels in the tree.

# Compiling and Testing

- The code for Bonsai is written in C++ and thus requires C++11 enabled compilers.
- To compile, run the `make` command from inside the `shallow` directory
- A sample script `sample_run.sh` is provided inside the `shallow` directory which trains the model and computes the metrics for the EUR-Lex dataset. 
- The usage of Bonsai for other datasets can be achieved from the above-mentioned `sample_run.sh` script by setting parameter `-m` to `2` for smaller datasets like EUR-Lex, Wikipedia-31K and to `3` for larger datasets like Delicious-200K, WikiLSHTC-325K, Amazon-670K, Wikipedia-500K, Amazon-3M.

# Miscellaneous

- A script `convert_data.sh` is provided in the `sandbox` directory to convert the raw train/test datasets from the XMC repository to the format needed by Bonsai.
- Scripts for performance evaluation are only available in Matlab. To compile these scripts, execute `make` in the tools/matlab folder from the Matlab terminal.

# Acknowledgement

The code is adapted from the source code kindly provided by the authors of [Parabel: Partitioned Label Trees for Extreme Classification with Application to Dynamic Search Advertising](https://dl.acm.org/citation.cfm?id=3185998)
