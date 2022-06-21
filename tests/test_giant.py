import sys

import pandas as pd
import pytest

if not sys.platform.startswith("linux"):
    # it's not necessary to test the REST API against GIANT in all platforms
    pytest.skip(
        "Skipping REST test on GIANT in non-Linux systems", allow_module_level=True
    )

from ccc.giant import gene_exists, predict_tissue, get_network


# Gene mappings used in unit tests
gene_mappings = pd.DataFrame(
    [
        ("IFNG", "3458"),
        ("SDS", "10993"),
        ("NR4A3", "8013"),
        ("UPK3B", "105375355"),
        ("GLIPR1", "11010"),
        ("ZDHHC12", "84885"),
        ("CCL18", "6362"),
    ],
    columns=["SYMBOL", "ENTREZID"],
)


def test_gene_exists_gene_does_exist():
    assert gene_exists(3458)


def test_gene_exists_gene_doest_not_exist():
    assert not gene_exists(100129354)
    assert not gene_exists(000000)


def test_predict_tissue_gene_pair_exists():
    assert predict_tissue(("6903", "3458")) == (
        "nervous-system",
        "http://hb.flatironinstitute.org/api/integrations/nervous-system/",
    )


def test_predict_tissue_gene_pair_does_not_exists():
    assert predict_tissue(("100129354", "871")) is None


def test_predict_tissue_top_tissue_is_not_brenda():
    assert predict_tissue(("3458", "10993")) == (
        "natural-killer-cell",
        "http://hb.flatironinstitute.org/api/integrations/natural-killer-cell/",
    )


def test_get_network_parameters_not_provided():
    with pytest.raises(ValueError) as e:
        get_network()

    assert "no arguments" in str(e).lower()


def test_get_network_gene_mappings_not_provided():
    with pytest.raises(ValueError) as e:
        get_network(gene_symbols=("IFNG", "SDS"))

    assert "gene mappings" in str(e).lower()


def test_get_network_gene_mappings_is_invalid():
    with pytest.raises(ValueError) as e:
        get_network(
            gene_symbols=("IFNG", "SDS"), gene_ids_mappings=gene_mappings.iloc[:, 0]
        )

    assert "gene_ids_mappings" in str(e).lower()


def test_get_network_gene_mappings_with_invalid_columns():
    with pytest.raises(ValueError) as e:
        get_network(
            gene_symbols=("IFNG", "SDS"),
            gene_ids_mappings=gene_mappings.rename(columns={"SYMBOL": "symbol"}),
        )

    assert "SYMBOL and ENTREZID" in str(e)


def test_get_network_genes_do_not_exist():
    # this case is wrong because gene symbols are given as entrezids
    assert get_network(("IFNG", "SDS"), gene_ids_mappings=gene_mappings) is None

    # one gene in the pair does not exist
    assert (
        get_network(
            gene_symbols=("IFNG", "nonExistingGene"), gene_ids_mappings=gene_mappings
        )
        is None
    )

    # in this case, UPK3B is not included in the returned network
    assert (
        get_network(gene_symbols=("NR4A3", "UPK3B"), gene_ids_mappings=gene_mappings)
        is None
    )


def test_get_network_ifng_glipr1():
    # these cases were obtained from the web application and are validated here

    # Go to https://hb.flatironinstitute.org/gene/3458+11010
    # and download the visible network as text file
    gene_symbols = ("IFNG", "GLIPR1")

    # Run
    res = get_network(gene_symbols=gene_symbols, gene_ids_mappings=gene_mappings)
    assert res is not None

    df, df_tissue, _ = res
    df = df.round(4)
    assert df.shape[0] == 134

    assert df_tissue == "blood"

    pd.testing.assert_series_equal(
        df.iloc[0],
        pd.Series(["HLA-DPA1", "GBP2", 0.8386]),
        check_names=False,
        check_index=False,
    )

    pd.testing.assert_series_equal(
        df.iloc[54],
        pd.Series(["LCP2", "CASP1", 0.7856]),
        check_names=False,
        check_index=False,
    )

    pd.testing.assert_series_equal(
        df.iloc[-1],
        pd.Series(["ITGB2", "HLA-DQB1", 0.8782]),
        check_names=False,
        check_index=False,
    )


def test_get_network_ifng_glipr1_with_entrezids():
    # these cases were obtained from the web application and are validated here

    # Go to https://hb.flatironinstitute.org/gene/3458+11010
    # and download the visible network as text file
    gene_entrezids = ("3458", "11010")

    # Run
    res = get_network(gene_entrezids=gene_entrezids, gene_ids_mappings=gene_mappings)
    assert res is not None

    df, df_tissue, _ = res
    df = df.round(4)
    assert df.shape[0] == 134

    assert df_tissue == "blood"

    pd.testing.assert_series_equal(
        df.iloc[0],
        pd.Series(["HLA-DPA1", "GBP2", 0.8386]),
        check_names=False,
        check_index=False,
    )

    pd.testing.assert_series_equal(
        df.iloc[54],
        pd.Series(["LCP2", "CASP1", 0.7856]),
        check_names=False,
        check_index=False,
    )

    pd.testing.assert_series_equal(
        df.iloc[-1],
        pd.Series(["ITGB2", "HLA-DQB1", 0.8782]),
        check_names=False,
        check_index=False,
    )


def test_get_network_zdhhc12_ccl18():
    # these cases were obtained from the web application and are validated here

    # Go to https://hb.flatironinstitute.org/gene/84885+6362
    # and download the visible network as text file
    gene_symbols = ("ZDHHC12", "CCL18")

    # Run
    res = get_network(gene_symbols=gene_symbols, gene_ids_mappings=gene_mappings)
    assert res is not None

    df, df_tissue, _ = res
    df = df.round(4)
    assert df.shape[0] == 129

    assert df_tissue == "macrophage"

    pd.testing.assert_series_equal(
        df.iloc[0],
        pd.Series(["CCL3", "SCAMP2", 0.1110]),
        check_names=False,
        check_index=False,
    )

    pd.testing.assert_series_equal(
        df.iloc[72],
        pd.Series(["ZDHHC12", "CTSB", 0.1667]),
        check_names=False,
        check_index=False,
    )

    pd.testing.assert_series_equal(
        df.iloc[-1],
        pd.Series(["C1QA", "HLA-DQB1", 0.4485]),
        check_names=False,
        check_index=False,
    )


def test_get_network_zdhhc12_ccl18_specify_tissue_blood():
    # these cases were obtained from the web application and are validated here

    # Go to https://hb.flatironinstitute.org/gene/84885+6362
    # select blood as tissue and download the visible network as text file
    gene_symbols = ("ZDHHC12", "CCL18")

    # Run
    blood_tissue_info = (
        "blood",
        "http://hb.flatironinstitute.org/api/integrations/blood/",
    )
    res = get_network(
        gene_symbols=gene_symbols,
        gene_ids_mappings=gene_mappings,
        tissue=blood_tissue_info,
    )
    assert res is not None

    df, df_tissue, _ = res
    df = df.round(4)
    assert df.shape[0] == 128

    assert df_tissue == "blood"

    pd.testing.assert_series_equal(
        df.iloc[0],
        pd.Series(["HLA-DRA", "MMP9", 0.5582]),
        check_names=False,
        check_index=False,
    )

    pd.testing.assert_series_equal(
        df.iloc[111],
        pd.Series(["GZMB", "C3AR1", 0.6288]),
        check_names=False,
        check_index=False,
    )

    pd.testing.assert_series_equal(
        df.iloc[-1],
        pd.Series(["C3AR1", "C1QA", 0.8413]),
        check_names=False,
        check_index=False,
    )


def test_get_network_zdhhc12_ccl18_with_invalid_tissue():
    # these cases were obtained from the web application and are validated here

    # Go to https://hb.flatironinstitute.org/gene/84885+6362
    # select blood as tissue and download the visible network as text file
    gene_symbols = ("ZDHHC12", "CCL18")

    # Run
    # invalid tissue info
    blood_tissue_info = ("blood",)

    with pytest.raises(ValueError) as e:
        get_network(
            gene_symbols=gene_symbols,
            gene_ids_mappings=gene_mappings,
            tissue=blood_tissue_info,
        )

    assert "invalid tissue" in str(e).lower()
