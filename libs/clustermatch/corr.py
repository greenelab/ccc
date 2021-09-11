"""
Functions to compute different correlation coefficients.

All correlation functions in this module are expected to have the same input and output
structure:

 * The input is a pandas DataFrame with genes in rows (Ensembl IDs) and samples
   columns. The values are gene expression data normalized with some technique,
   but that should not be relevant for the correlation method.

 * The output is a pandas DataFrame, a symmetric correlation matrix with genes
   in rows and columns (Ensembl IDs), and the values are the correlation
   coefficients. Diagonal values are expected to be ones.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances


def pearson(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pearson correlation coefficient.
    """
    corr_mat = 1 - pairwise_distances(data.to_numpy(), metric="correlation", n_jobs=1)

    np.fill_diagonal(corr_mat, 1.0)

    return pd.DataFrame(
        corr_mat,
        index=data.index.copy(),
        columns=data.index.copy(),
    )


def spearman(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Spearman correlation coefficient.
    """
    # compute ranks
    data = data.rank(axis=1)

    corr_mat = 1 - pairwise_distances(data.to_numpy(), metric="correlation", n_jobs=1)

    np.fill_diagonal(corr_mat, 1.0)

    return pd.DataFrame(
        corr_mat,
        index=data.index.copy(),
        columns=data.index.copy(),
    )


def clustermatch_naive(data: pd.DataFrame) -> pd.DataFrame:
    """
    TODO: it would be good to have this one to compare performance with optmized version

    - do not use this; use instead clustermatch
    """
    from clustermatch.coef import cm

    corr_mat = pairwise_distances(
        data.to_numpy(), metric=cm, n_jobs=1, force_all_finite="allow-nan"
    )

    np.fill_diagonal(corr_mat, 1.0)

    return pd.DataFrame(
        corr_mat,
        index=data.index.copy(),
        columns=data.index.copy(),
    )


def clustermatch(data: pd.DataFrame) -> pd.DataFrame:
    pass
