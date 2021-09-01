"""
Functions to compute different correlation coefficients.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances


def pearson(data):
    corr_mat = 1 - pairwise_distances(data.to_numpy(), metric="correlation", n_jobs=1)

    np.fill_diagonal(corr_mat, 1.0)

    return pd.DataFrame(
        corr_mat,
        index=data.index.copy(),
        columns=data.index.copy(),
    )


def spearman(data):
    # compute ranks
    data = data.rank(axis=1)

    corr_mat = 1 - pairwise_distances(data.to_numpy(), metric="correlation", n_jobs=1)

    np.fill_diagonal(corr_mat, 1.0)

    return pd.DataFrame(
        corr_mat,
        index=data.index.copy(),
        columns=data.index.copy(),
    )
