"""
Tests the corr.py module.
"""
import numpy as np
import pandas as pd

from clustermatch import corr


def _get_random_data(n_genes, n_samples, random_state=0):
    np.random.seed(random_state)

    # simulate data with a real structure of genes and samples
    random_data = np.random.rand(n_genes, n_samples)
    return pd.DataFrame(
        data=random_data,
        index=[f"ENSG00000123456.{i}" for i in range(random_data.shape[0])],
        columns=[f"Sample-Number-{i}" for i in range(random_data.shape[1])],
    )


def _run_basic_checks(corr_method, random_state=0):
    n_genes = 10
    n_samples = 100
    random_data = _get_random_data(n_genes, n_samples, random_state)

    # run
    corr_mat = corr_method(random_data)
    assert corr_mat is not None

    # shape
    assert hasattr(corr_mat, "shape")
    assert corr_mat.shape == (n_genes, n_genes)

    # row/columns names are genes
    assert hasattr(corr_mat, "index")
    assert list(corr_mat.index) == list(random_data.index)

    assert hasattr(corr_mat, "columns")
    assert list(corr_mat.columns) == list(random_data.index)

    assert hasattr(corr_mat, "columns")

    # dtype
    assert np.issubdtype(corr_mat.to_numpy().dtype, np.number)

    # diagonal has ones
    assert np.array_equal(np.diag(corr_mat), np.ones(n_genes))

    # matrix is symmetric
    assert corr_mat.equals(corr_mat.T)

    return random_data, corr_mat


def test_corr_pearson():
    # run basic tests first
    data, corr_mat = _run_basic_checks(corr.pearson)

    corr_values = pd.Series(corr_mat.to_numpy().flatten())

    # check ranges
    assert corr_values.max() <= 1.0
    assert corr_values.min() >= -1.0
    assert np.sign(corr_values.max()) != np.sign(corr_values.min())

    # calculate pearson with a different method and check if it is the same
    numpy_pearson = np.corrcoef(data.to_numpy())

    assert np.allclose(
        numpy_pearson,
        corr_mat.to_numpy(),
    )
