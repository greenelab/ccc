"""
Tests the corr.py module.
"""
import numpy as np
import pandas as pd

from clustermatch import corr


def _get_random_data(n_genes: int, n_samples: int, random_state=0) -> pd.DataFrame:
    """
    Simulates with random data a gene expression data matrix in the same format
    of real datasets.

    Args:
        n_genes: number of genes (rows) to be generated.
        n_samples: number of samples (columns) to be generated.
        random_state: random seed for np.random.seed.

    Returns:
        A pandas DataFrame with random numerical values generated with
        np.random.rand. The index will have simulated gene Ensembl IDs with the
        following format: ENSG00000123456.{i}, where {i} is the index of the
        gene (starting from zero to n_genes - 1). The columns will have
        simulated sample IDs with the following format: Sample-Number-{i}.
    """
    np.random.seed(random_state)

    # simulate data with a real structure of genes and samples
    random_data = np.random.rand(n_genes, n_samples)

    return pd.DataFrame(
        data=random_data,
        index=[f"ENSG00000123456.{i}" for i in range(random_data.shape[0])],
        columns=[f"Sample-Number-{i}" for i in range(random_data.shape[1])],
    )


def _run_basic_checks(corr_method, random_state=0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs basic checks on the output of a correlation method.

    Args:
        corr_method: a function that computes the correlations among genes. This
            function receives the data as the only arguments, which has the same
            format returned by function _get_random_data. It must return a
            correlation matrix as with the same format specified in the corr.py
            module (description at the top of file).
        random_state: passed to the _get_random_data function.

    Returns:
        A tuple with the random data generated and the correlation matrix.
    """
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
    numpy_pearson_mat = np.corrcoef(data.to_numpy())

    assert np.allclose(
        numpy_pearson_mat,
        corr_mat.to_numpy(),
    )


def test_corr_spearman():
    # run basic tests first
    data, corr_mat = _run_basic_checks(corr.spearman)

    corr_values = pd.Series(corr_mat.to_numpy().flatten())

    # check ranges
    assert corr_values.max() <= 1.0
    assert corr_values.min() >= -1.0
    assert np.sign(corr_values.max()) != np.sign(corr_values.min())

    # calculate pearson with a different method and check if it is the same
    from scipy.stats import spearmanr

    scipy_spearman_mat = spearmanr(data.to_numpy(), axis=1)[0]

    assert np.allclose(
        scipy_spearman_mat,
        corr_mat.to_numpy(),
    )
