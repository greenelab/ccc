"""
TODO
"""
import warnings

import pandas as pd
from minepy.mine import MINE


def mic(x, y):
    """
    Given two arrays (x and y), it computes MIC with the default parameters.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        mine = MINE(alpha=0.6, c=15, est="mic_approx")
        mine.compute_score(x, y)
        return mine.mic()


def _compute_mic(gene_sets: list, gene_expr_dict):
    """
    It takes a list of gene pairs and computes MIC on all.
    It returns a series with gene pairs as index and MIC values.
    This function is used in concurrent.futures for parallel execution.
    """
    res = [
        mic(gene_expr_dict[gs[0]].to_numpy(), gene_expr_dict[gs[1]].to_numpy())
        for gs in gene_sets
    ]

    return pd.Series(res, index=pd.MultiIndex.from_tuples(gene_sets))
