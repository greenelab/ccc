"""
Tests the corr.py module.
"""
import numpy as np
import pandas as pd

from ccc import corr


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


def test_corr_pearson_manual():
    # add basic check with manual calculation of the correlation
    x = np.array([0, 1, 2, 3])
    y = np.array([0, -1, -3, 8])
    test_data = pd.DataFrame(np.array([x, y]))

    num = (x - x.mean()) @ (y - y.mean())
    dem = np.sqrt(np.sum((x - x.mean()) ** 2) * np.sum((y - y.mean()) ** 2))
    expected_corr = num / dem
    assert expected_corr == 0.5879747322073337

    test_result = corr.pearson(test_data)
    assert test_result.iloc[0, 0] == 1.0
    assert test_result.iloc[0, 1] == expected_corr
    assert test_result.iloc[1, 0] == expected_corr
    assert test_result.iloc[1, 1] == 1.0


def test_corr_spearman():
    # run basic tests first
    data, corr_mat = _run_basic_checks(corr.spearman)

    corr_values = pd.Series(corr_mat.to_numpy().flatten())

    # check ranges
    assert corr_values.max() <= 1.0
    assert corr_values.min() >= -1.0
    assert np.sign(corr_values.max()) != np.sign(corr_values.min())

    # calculate spearman with a different method and check if it is the same
    from scipy.stats import spearmanr

    scipy_spearman_mat = spearmanr(data.to_numpy(), axis=1)[0]

    assert np.allclose(
        scipy_spearman_mat,
        corr_mat.to_numpy(),
    )


def test_corr_spearman_manual():
    # add basic check with manual calculation of the correlation
    x = np.array([0, 1, 2, 3])
    y = np.array([0, -1, -3, 8])
    test_data = pd.DataFrame(np.array([x, y]))

    # get ranks
    order = x.argsort()
    x = order.argsort()

    order = y.argsort()
    y = order.argsort()

    num = (x - x.mean()) @ (y - y.mean())
    dem = np.sqrt(np.sum((x - x.mean()) ** 2) * np.sum((y - y.mean()) ** 2))
    expected_corr = num / dem
    assert round(expected_corr, 5) == 0.2

    test_result = corr.spearman(test_data)
    assert test_result.iloc[0, 0] == 1.0
    assert test_result.iloc[0, 1].round(5) == expected_corr
    assert test_result.iloc[1, 0].round(5) == expected_corr
    assert test_result.iloc[1, 1] == 1.0


def test_corr_mic():
    # run basic tests first
    data, corr_mat = _run_basic_checks(corr.mic)

    corr_values = pd.Series(corr_mat.to_numpy().flatten())

    # check ranges
    assert corr_values.max() <= 1.0
    assert corr_values.min() >= 0.0
    assert np.sign(corr_values.max()) == np.sign(corr_values.min())


def test_corr_mic_manual():
    # add basic check with manual calculation of the correlation
    x = np.array([0, 1, 2, 3])
    y = np.array([0, -1, -3, 8])
    test_data = pd.DataFrame(np.array([x, y]))

    # compute original mic
    from minepy.mine import MINE

    mine = MINE(alpha=0.6, c=15, est="mic_approx")
    mine.compute_score(x, y)
    expected_corr = mine.mic()
    assert round(expected_corr, 5) == 0.31128

    test_result = corr.mic(test_data)
    assert test_result.iloc[0, 0] == 1.0
    assert test_result.iloc[0, 1].round(5) == round(expected_corr, 5)
    assert test_result.iloc[1, 0].round(5) == round(expected_corr, 5)
    assert test_result.iloc[1, 1] == 1.0


def test_corr_mic_parallel():
    # run basic tests first
    data, corr_mat = _run_basic_checks(lambda data: corr.mic(data, n_jobs=2))

    corr_values = pd.Series(corr_mat.to_numpy().flatten())

    # check ranges
    assert corr_values.max() <= 1.0
    assert corr_values.min() >= 0.0
    assert np.sign(corr_values.max()) == np.sign(corr_values.min())


def test_corr_mic_parallel_manual():
    # add basic check with manual calculation of the correlation
    x = np.array([0, 1, 2, 3])
    y = np.array([0, -1, -3, 8])
    test_data = pd.DataFrame(np.array([x, y]))

    # compute original mic
    from minepy.mine import MINE

    mine = MINE(alpha=0.6, c=15, est="mic_approx")
    mine.compute_score(x, y)
    expected_corr = mine.mic()
    assert round(expected_corr, 5) == 0.31128

    test_result = corr.mic(test_data, n_jobs=2)
    assert test_result.iloc[0, 0] == 1.0
    assert test_result.iloc[0, 1].round(5) == round(expected_corr, 5)
    assert test_result.iloc[1, 0].round(5) == round(expected_corr, 5)
    assert test_result.iloc[1, 1] == 1.0


def test_corr_clustermatch_basics():
    # run basic tests first
    data, corr_mat = _run_basic_checks(corr.ccc)

    corr_values = pd.Series(corr_mat.to_numpy().flatten())

    # check ranges
    assert corr_values.max() <= 1.0
    assert corr_values.min() >= 0.0
    assert np.sign(corr_values.max()) == np.sign(corr_values.min())


def test_corr_clustermatch_outputs_same_as_original_clustermatch():
    # compare with results obtained from the original ccc
    # implementation (https://github.com/sinc-lab/clustermatch) plus some
    # patches (see README.md in tests/data about ccc data).
    from pathlib import Path
    from pandas.testing import assert_frame_equal

    input_data_dir = Path(__file__).parent / "data"

    # load data
    data = pd.read_pickle(input_data_dir / "ccc-random_data-data.pkl")

    # run new ccc implementation.
    # Here, I fixed the internal number of clusters, since that slightly changed
    # in the new implementation compared with the original one.
    corr_mat = corr.ccc(data, internal_n_clusters=list(range(2, 10 + 1)))

    expected_corr_matrix = pd.read_pickle(input_data_dir / "ccc-random_data-coef.pkl")
    expected_corr_matrix = expected_corr_matrix.loc[data.index, data.index]

    assert corr_mat.shape == expected_corr_matrix.shape
    assert corr_mat.index.tolist() == expected_corr_matrix.index.tolist()
    assert corr_mat.columns.tolist() == expected_corr_matrix.columns.tolist()

    assert_frame_equal(
        expected_corr_matrix,
        corr_mat,
        check_exact=False,
    )
