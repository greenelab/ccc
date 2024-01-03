# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# It plots specific gene pairs from different subsets of the UpSet plot (intersections) generated before.
#
# The idea of the notebook is to take a look at the patterns found/not found by different methods.
#
# **Note after manuscript revision:** The "Selection" section below each gene pair category has a selection considering the first selection (first manuscript submission) and a second selection (after manuscript revision) considering the p-value.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import pandas as pd
import numpy as np
import pytest

import matplotlib.pyplot as plt
import seaborn as sns

from ccc.plots import jointplot
from ccc import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
DATASET_CONFIG = conf.GTEX
GTEX_TISSUE = "whole_blood"
GENE_SEL_STRATEGY = "var_pc_log2"

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
assert (
    conf.MANUSCRIPT["BASE_DIR"] is not None and conf.MANUSCRIPT["BASE_DIR"].exists()
), "Manuscript dir not set"

# %% tags=[]
OUTPUT_DATA_DIR = DATASET_CONFIG["GENE_PAIR_INTERSECTIONS"]
assert OUTPUT_DATA_DIR.exists()
display(OUTPUT_DATA_DIR)

# %% tags=[]
OUTPUT_FIGURE_DIR = (
    conf.MANUSCRIPT["FIGURES_DIR"] / "coefs_comp" / f"gtex_{GTEX_TISSUE}"
)
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_FIGURE_DIR)

# %% tags=[]
INPUT_GENE_EXPR_FILE = (
    DATASET_CONFIG["GENE_SELECTION_DIR"]
    / f"gtex_v8_data_{GTEX_TISSUE}-{GENE_SEL_STRATEGY}.pkl"
)
display(INPUT_GENE_EXPR_FILE)

assert INPUT_GENE_EXPR_FILE.exists()

# %% tags=[]
INPUT_GENE_PAIRS_INTERSECTIONS_FILE = (
    DATASET_CONFIG["GENE_PAIR_INTERSECTIONS"]
    / f"gene_pair_intersections-gtex_v8-{GTEX_TISSUE}-{GENE_SEL_STRATEGY}.pkl"
)
display(INPUT_GENE_PAIRS_INTERSECTIONS_FILE)

assert INPUT_GENE_PAIRS_INTERSECTIONS_FILE.exists()

# %% tags=[]
INPUT_PVALUES_FILE = (
    DATASET_CONFIG["GENE_PAIR_INTERSECTIONS"]
    / "pvalues"
    / "gene_pair-samples-pvalues-fdr.pkl"
)
display(INPUT_PVALUES_FILE)
assert INPUT_PVALUES_FILE.exists()

# %% [markdown] tags=[]
# # Data

# %% [markdown] tags=[]
# ## Gene Ensembl ID -> Symbol mapping

# %% tags=[]
gene_map = pd.read_pickle(
    DATASET_CONFIG["DATA_DIR"] / "gtex_gene_id_symbol_mappings.pkl"
)

# %% tags=[]
gene_map = gene_map.set_index("gene_ens_id")["gene_symbol"].to_dict()

# %% tags=[]
assert gene_map["ENSG00000145309.5"] == "CABS1"

# %% [markdown] tags=[]
# ## Gene expression

# %% tags=[]
gene_expr_df = pd.read_pickle(INPUT_GENE_EXPR_FILE)

# %% tags=[]
gene_expr_df.shape

# %% tags=[]
gene_expr_df.head()

# %% [markdown] tags=[]
# ## Gene pairs intersection

# %% tags=[]
df_plot = pd.read_pickle(INPUT_GENE_PAIRS_INTERSECTIONS_FILE)

# %% tags=[]
df_plot.shape

# %% tags=[]
df_plot.head()

# %% [markdown] tags=[]
# ## p-values

# %% tags=[]
df_pvalues = pd.read_pickle(INPUT_PVALUES_FILE).sort_index()

# %% tags=[]
df_pvalues.shape

# %% tags=[]
df_pvalues.head()

# %% tags=[]
# remove duplicated gene pairs
df_pvalues = df_pvalues[~df_pvalues.index.duplicated(keep="first")]

# %% [markdown] tags=[]
# **Note**: Here the "group" column specifies the categories in Figure 3a, followed by `top_[coef]`, where for the same category I sorted gene pairs by `coef`. This allows me, for instance, to take the gene pairs where Pearson is high and CCC is low, and sort by any of those coefficient values.

# %% [markdown] tags=[]
# ## Understudied list of genes

# %% [markdown] tags=[]
# ### Gene ID maps

# %% tags=[]
gene_n_papers = pd.read_pickle(
    conf.UNDERSTUDIED_GENES_ARTICLE["DATA_DIR"] / "s3_table.pkl"
)

# %% tags=[]
gene_n_papers.shape

# %% tags=[]
gene_n_papers.head()

# %% tags=[]
gene_ncbi_map = (
    gene_n_papers.reset_index()[["symbol_ncbi", "gene_ncbi"]]
    .set_index("symbol_ncbi")
    .squeeze()
    .to_dict()
)

# %% tags=[]
assert gene_ncbi_map["SDS"] == 10993

# %% [markdown] tags=[]
# ### Genes' predicted publications

# %% tags=[]
genes_predicted_pubs = pd.read_pickle(
    conf.UNDERSTUDIED_GENES_ARTICLE["DATA_DIR"] / "s1_data_1a.pkl"
).sort_values("diff", ascending=False)

# %% tags=[]
genes_predicted_pubs.shape

# %% tags=[]
genes_predicted_pubs.head()

# %% [markdown] tags=[]
# ## Gene pairs percentiles

# %% tags=[]
df_plot_percentiles = df_plot[["ccc", "pearson", "spearman"]].quantile(
    np.arange(0.1, 1.01, 0.01)
)

# %% tags=[]
with pd.option_context("display.max_rows", None):
    display(df_plot_percentiles)

# %% [markdown] tags=[]
# # Look at specific gene pair cases

# %% tags=[]
# add columns with ranks
df_r_data = pd.concat(
    [
        df_plot,
        df_plot[["ccc", "pearson", "spearman"]]
        .rank()
        .rename(
            columns={
                "ccc": "clustermatch_rank",
                "pearson": "pearson_rank",
                "spearman": "spearman_rank",
            }
        ),
    ],
    axis=1,
)

# %% tags=[]
df_r_data.shape

# %% tags=[]
df_r_data.head()

# %% tags=[]
# add p-values
df_r_data = df_r_data.join(
    df_pvalues.rename_axis(index=(None, None))[
        ["ccc_fdr", "pearson_fdr", "spearman_fdr"]
    ],
    how="left",
)

# %% tags=[]
df_r_data.shape

# %% tags=[]
df_r_data.head()

# %% tags=[]
df_r_data_boolean_cols = set(
    [x for x in df_r_data.columns if " (high)" in x or " (low)" in x]
)

# %% tags=[]
df_r_data_boolean_cols


# %% [markdown] tags=[]
# ## Functions

# %% tags=[]
def get_understudied_score_for_row(x) -> float:
    """
    Returns the maximum difference between "predicted" and "target" for a gene pair.
    """
    gene_pair_ens_ids = x.name
    gene_pair_symbols = gene_map[gene_pair_ens_ids[0]], gene_map[gene_pair_ens_ids[1]]
    gene_pair_ncbi_ids = gene_ncbi_map.get(
        gene_pair_symbols[0], None
    ), gene_ncbi_map.get(gene_pair_symbols[1], None)

    max_score = -np.inf

    if (
        gene_pair_ncbi_ids[0] is not None
        and gene_pair_ncbi_ids[0] in genes_predicted_pubs.index
    ):
        new_score = genes_predicted_pubs.loc[gene_pair_ncbi_ids[0], "diff"]
        if new_score > max_score:
            max_score = new_score

    if (
        gene_pair_ncbi_ids[1] is not None
        and gene_pair_ncbi_ids[1] in genes_predicted_pubs.index
    ):
        new_score = genes_predicted_pubs.loc[gene_pair_ncbi_ids[1], "diff"]
        if new_score > max_score:
            max_score = new_score

    return max_score


# %% tags=[]
def get_min_n_pubs(x) -> float:
    """
    Returns the minimum number of publications for a gene pair.
    """
    gene_pair_ens_ids = x.name
    gene_pair_symbols = gene_map[gene_pair_ens_ids[0]], gene_map[gene_pair_ens_ids[1]]
    gene_pair_ncbi_ids = gene_ncbi_map.get(
        gene_pair_symbols[0], None
    ), gene_ncbi_map.get(gene_pair_symbols[1], None)

    min_n_pubs = np.inf

    if (
        gene_pair_ncbi_ids[0] is not None
        and gene_pair_ncbi_ids[0] in gene_n_papers.index
    ):
        n_pubs = gene_n_papers.loc[gene_pair_ncbi_ids[0], "papers"]
        if n_pubs < min_n_pubs:
            min_n_pubs = n_pubs

    if (
        gene_pair_ncbi_ids[1] is not None
        and gene_pair_ncbi_ids[1] in gene_n_papers.index
    ):
        n_pubs = gene_n_papers.loc[gene_pair_ncbi_ids[1], "papers"]
        if n_pubs < min_n_pubs:
            min_n_pubs = n_pubs

    return min_n_pubs


# %% tags=[]
def plot_gene_pair(top_pairs_df, idx, bins="log", plot_gene_ids=True):
    """
    It plots a gene pair using a hexbin plot. The idea of this function is
    to quickly have an idea of the patterns (if any) present in a couple genes.

    Args:
        top_pairs_df: a dataframe with a preselected group of genes pairs (for instance,
            those where pearson is high and ccc is low. Each row is a gene pair.
            It is the output of function get_gene_pairs.
        idx: an integer that indicates which row in top_pairs_df you want to plot.
        bins: the "bins" parameter of seaborn's jointplot.
        plot_gene_ids: it adds genes' Ensembl IDs to the x and y labels.

    Returns:
        The JointGrid object returned by seaborn.jointplot.
    """
    gene0, gene1 = top_pairs_df.iloc[idx].name
    display((gene0, gene1))

    gene0_symbol, gene1_symbol = gene_map[gene0], gene_map[gene1]
    display((gene0_symbol, gene1_symbol))

    if "ccc_fdr" in top_pairs_df.columns:
        (
            _pearson,
            _pearson_fdr,
            _spearman,
            _spearman_fdr,
            _ccc,
            _ccc_fdr,
        ) = top_pairs_df.loc[
            (gene0, gene1),
            ["pearson", "pearson_fdr", "spearman", "spearman_fdr", "ccc", "ccc_fdr"],
        ].tolist()
    else:
        _pearson, _spearman, _ccc = top_pairs_df.loc[
            (gene0, gene1), ["pearson", "spearman", "ccc"]
        ].tolist()

    p = sns.jointplot(
        data=gene_expr_df.T,
        x=gene0,
        y=gene1,
        kind="hex",
        bins=bins,
        # ylim=(0, 500),
    )

    gene_x_id = p.ax_joint.get_xlabel()
    gene_x_symbol = gene_map[gene_x_id]

    gene_y_id = p.ax_joint.get_ylabel()
    gene_y_symbol = gene_map[gene_y_id]

    if plot_gene_ids:
        p.ax_joint.set_xlabel(f"{gene_x_id}\n{gene_x_symbol}")
        p.ax_joint.set_ylabel(f"{gene_y_id}\n{gene_y_symbol}")
    else:
        p.ax_joint.set_xlabel(f"{gene_x_symbol}", fontstyle="italic")
        p.ax_joint.set_ylabel(f"{gene_y_symbol}", fontstyle="italic")

    if "ccc_fdr" in top_pairs_df.columns:
        _title = f"$c={_ccc:.2f}$ (${_ccc_fdr:.2e}$)  $p={_pearson:.2f}$ (${_pearson_fdr:.2e}$)    $s={_spearman:.2f}$ (${_spearman_fdr:.2e}$)"
    else:
        _title = f"$c={_ccc:.2f}$  $p={_pearson:.2f}$    $s={_spearman:.2f}$"

    understudied_score, min_n_pubs = top_pairs_df.loc[
        (gene0, gene1), ["understudied_score", "min_n_pubs"]
    ]
    _title += f"\nunderstudied_score={understudied_score:.2f}  min_n_pubs={min_n_pubs}"

    p.fig.suptitle(_title)

    return p


# %% tags=[]
def get_gene_pairs(first_coef, query_set):
    """
    It queries a dataframe with the intersections of different groups (i.e.,
    ccc high, pearson low, etc) given a query set. It returns a slice of
    the dataframe according to the query set provided.

    The function needs to access a variable named "df_r_data" that has the
    intersections between coefficients.

    Args:
        first_coef: the main coefficient ("ccc", "pearson" or "spearman")
            of interest. The final dataframe will be sorted according to this
            coefficient.
        query_set: a tuple with strings that specifies a query. For example
            ("Clustermatch (high)", "Pearson (low") would select all gene pairs
            for which ccc is high and pearson is low.

    Returns:
        A slice of variable "data_r_data" where the conditions specified in query_set
        apply.
    """
    assert all([x in df_r_data_boolean_cols for x in query_set])

    query = np.concatenate(
        [
            # columns that have to be true
            np.concatenate(
                [df_r_data[c].to_numpy().reshape(-1, 1) for c in query_set], axis=1
            )
            .all(axis=1)
            .reshape(-1, 1),
            # rest of the columns, that have to be false
            np.concatenate(
                [
                    ~df_r_data[c].to_numpy().reshape(-1, 1)
                    for c in df_r_data_boolean_cols
                    if c not in query_set
                ],
                axis=1,
            )
            .all(axis=1)
            .reshape(-1, 1),
        ],
        axis=1,
    ).all(axis=1)

    _tmp_df = df_r_data[query]

    # sort by firt_coef value
    _tmp_df = _tmp_df.sort_values(first_coef, ascending=False)

    _tmp_df = _tmp_df[
        [x for x in _tmp_df.columns if "(high)" not in x and "(low)" not in x]
    ]

    # add understudied_score
    _tmp_df = _tmp_df.assign(
        understudied_score=_tmp_df.apply(get_understudied_score_for_row, axis=1)
    )

    # add minimum number of publications
    _tmp_df = _tmp_df.assign(min_n_pubs=_tmp_df.apply(get_min_n_pubs, axis=1))

    return _tmp_df


# %% tags=[]
def plot_and_save_gene_pair(data, gene0_id, gene1_id, output_file_subset):
    """
    This function creates a joint plot of a pair of genes. It's used to to
    select the final gene pairs to include in the paper.
    """
    gene0_symbol = gene_map[gene0_id]
    gene1_symbol = gene_map[gene1_id]

    with sns.plotting_context("paper", font_scale=2.0):
        p = jointplot(
            data,
            x=gene0_id,
            y=gene1_id,
            add_corr_coefs=False,
        )

        p.ax_joint.set_xlabel(f"{gene0_symbol}", fontstyle="italic")
        p.ax_joint.set_ylabel(f"{gene1_symbol}", fontstyle="italic")

        output_file = (
            OUTPUT_FIGURE_DIR
            / f"genes-{output_file_subset}-{gene0_symbol}_vs_{gene1_symbol}.svg"
        )
        display(output_file)

        plt.savefig(
            output_file,
            bbox_inches="tight",
            dpi=300,
            facecolor="white",
        )


# %% tags=[]
def save_gene_pairs(df, gene_set_name):
    """
    Given a dataframe with gene pairs (prioritized by one correlation coefficient over the other coefficients)
    and a gene set name, it simply saves the dataframe into a file. It also rename gene ensemble IDs to symbols.
    """
    # convert gene ids to gene symbols
    df = (
        df.reset_index()
        .replace(
            {
                "level_0": gene_map,
                "level_1": gene_map,
            }
        )
        .set_index(["level_0", "level_1"])
        .rename_axis([None, None])
    )

    df.to_pickle(OUTPUT_DATA_DIR / f"{gene_set_name}.pkl")


# %% tags=[]
def get_percentiles(row):
    """
    Helper function that given a row/dict with columns "ccc", "pearson" and "spearman" correlation values,
    it returns under which percentile they are.
    """
    _ccc = row["ccc"]
    _p = row["pearson"]
    _s = row["spearman"]

    return {
        "ccc": df_plot_percentiles["ccc"].ge(_ccc).idxmax(),
        "pearson": df_plot_percentiles["pearson"].ge(_p).idxmax(),
        "spearman": df_plot_percentiles["spearman"].ge(_s).idxmax(),
    }


# %% tags=[]
# testing
_tmp = get_percentiles({"ccc": 0.706993, "pearson": 0.090451, "spearman": 0.765177})
assert _tmp["ccc"] == pytest.approx(1.0)
assert _tmp["pearson"] == pytest.approx(0.25)
assert _tmp["spearman"] == pytest.approx(0.90)

_tmp = get_percentiles({"ccc": 0.013, "pearson": 0.9948, "spearman": 0.986})
assert _tmp["ccc"] == pytest.approx(0.1)
assert _tmp["pearson"] == pytest.approx(1.0)
assert _tmp["spearman"] == pytest.approx(1.0)

# %% [markdown] tags=[]
# ## CCC/Spearman vs Pearson

# %% tags=[]
_coef_fdr_col_name = "pearson_fdr"

# %% tags=[]
_tmp_df = get_gene_pairs(
    "ccc",
    {
        "Clustermatch (high)",
        "Spearman (high)",
        "Pearson (low)",
    },
)

save_gene_pairs(_tmp_df, "clustermatch_spearman_vs_pearson")

display(_tmp_df.shape)
display(_tmp_df)

# %% [markdown] tags=[]
# ### Preview sorting by top coefficient value only

# %% tags=[]
for i in range(min(_tmp_df.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ### Preview sorting by top coefficient value and non-significant other p-value

# %% tags=[]
# take a look considering the least significant p-values of the other coefficient
_tmp_df_pval = _tmp_df[_tmp_df[_coef_fdr_col_name] > 0.05]
display(_tmp_df_pval)

# %% tags=[]
for i in range(min(_tmp_df_pval.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df_pval, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ### Preview sorting non-significant other p-value

# %% tags=[]
# take a look considering the least significant p-values of the other coefficient
_tmp_df_other_pval = _tmp_df[_tmp_df[_coef_fdr_col_name] > 0.05].sort_values(
    _coef_fdr_col_name, ascending=False
)
display(_tmp_df_other_pval)

# %% tags=[]
for i in range(min(_tmp_df_other_pval.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df_other_pval, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ### Selection

# %% tags=[]
gene_pair_subset = "c_rs_vs_r"

# %% [markdown] tags=[]
# #### Initial selection
#
# Initial (first manuscript submission) gene pair selection considering coefficient values only (but not their p-value):

# %% tags=[]
gene0_id = "ENSG00000135094.10"
gene1_id = "ENSG00000111537.4"

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %% tags=[]
# get percentiles
_tmp = df_plot.loc[(gene0_id, gene1_id), ["ccc", "pearson", "spearman"]]
display(get_percentiles(_tmp))

# %% tags=[]
gene0_id = "ENSG00000130208.9"
gene1_id = "ENSG00000177606.6"

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %% tags=[]
# get percentiles
_tmp = df_plot.loc[(gene0_id, gene1_id), ["ccc", "pearson", "spearman"]]
display(get_percentiles(_tmp))

# %% [markdown] tags=[]
# #### Second selection
#
# Second (after first revision) gene pair selection, which considers both coefficient values and p-value.

# %% tags=[]
gene0_id = "ENSG00000130208.9"
gene1_id = "ENSG00000128342.4"

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %% tags=[]
# get percentiles
_tmp = df_plot.loc[(gene0_id, gene1_id), ["ccc", "pearson", "spearman"]]
display(get_percentiles(_tmp))

# %% [markdown] tags=[]
# ## CCC vs Pearson

# %% tags=[]
_coef_fdr_col_name = "pearson_fdr"

# %% tags=[]
_tmp_df = get_gene_pairs(
    "ccc",
    {
        "Clustermatch (high)",
        "Pearson (low)",
    },
)

save_gene_pairs(_tmp_df, "clustermatch_vs_pearson")

display(_tmp_df.shape)
display(_tmp_df)

# %% [markdown] tags=[]
# ### Preview sorting by top coefficient value only

# %% tags=[]
for i in range(min(_tmp_df.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ### Preview sorting by top coefficient value and non-significant other p-value

# %% tags=[]
# take a look considering the least significant p-values of the other coefficient
_tmp_df_pval = _tmp_df[
    (_tmp_df[_coef_fdr_col_name] > 0.05)
]  # .sort_values("spearman_fdr", ascending=False)
display(_tmp_df_pval)

# %% tags=[]
for i in range(min(_tmp_df_pval.shape[0], 10)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df_pval, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ### Preview sorting non-significant other p-value

# %% tags=[]
# take a look considering the least significant p-values of the other coefficient
_tmp_df_other_pval = _tmp_df[_tmp_df[_coef_fdr_col_name] > 0.05].sort_values(
    _coef_fdr_col_name, ascending=False
)
display(_tmp_df_other_pval)

# %% tags=[]
for i in range(min(_tmp_df_other_pval.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df_other_pval, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ### Selection

# %% tags=[]
gene_pair_subset = "c_vs_r"

# %% [markdown] tags=[]
# #### Initial selection
#
# Initial (first manuscript submission) gene pair selection considering coefficient values only (but not their p-value):

# %% tags=[]
gene0_id = "ENSG00000275385.1"
gene1_id = "ENSG00000160446.18"

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %% tags=[]
# get percentiles
_tmp = df_plot.loc[(gene0_id, gene1_id), ["ccc", "pearson", "spearman"]]
display(get_percentiles(_tmp))

# %% [markdown] tags=[]
# #### Second selection
#
# Second (after first revision) gene pair selection, which considers both coefficient values and p-value.

# %% tags=[]
gene0_id = "ENSG00000275385.1"
gene1_id = "ENSG00000178226.10"

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %% tags=[]
# get percentiles
_tmp = df_plot.loc[(gene0_id, gene1_id), ["ccc", "pearson", "spearman"]]
display(get_percentiles(_tmp))

# %% [markdown] tags=[]
# ## CCC vs Spearman

# %% tags=[]
_coef_fdr_col_name = "spearman_fdr"

# %% tags=[]
_tmp_df = get_gene_pairs(
    "ccc",
    {
        "Clustermatch (high)",
        "Spearman (low)",
    },
)

save_gene_pairs(_tmp_df, "clustermatch_vs_spearman")

display(_tmp_df.shape)
display(_tmp_df)

# %% [markdown] tags=[]
# ### Preview sorting by top coefficient value only

# %% tags=[]
for i in range(min(_tmp_df.shape[0], 30)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ### Preview sorting by top coefficient value and non-significant other p-value

# %% tags=[]
# take a look considering the least significant p-values of the other coefficient
_tmp_df_pval = _tmp_df[
    (_tmp_df[_coef_fdr_col_name] > 0.05)
]  # .sort_values("spearman_fdr", ascending=False)
display(_tmp_df_pval)

# %% tags=[]
for i in range(min(_tmp_df_pval.shape[0], 10)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df_pval, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ### Preview sorting non-significant other p-value

# %% tags=[]
# take a look considering the least significant p-values of the other coefficient
_tmp_df_other_pval = _tmp_df[_tmp_df[_coef_fdr_col_name] > 0.05].sort_values(
    _coef_fdr_col_name, ascending=False
)
display(_tmp_df_other_pval)

# %% tags=[]
for i in range(min(_tmp_df_other_pval.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df_other_pval, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ### Selection

# %% tags=[]
gene_pair_subset = "c_vs_rs"

# %% [markdown] tags=[]
# #### Initial selection
#
# Initial (first manuscript submission) gene pair selection considering coefficient values only (but not their p-value):

# %% tags=[]
gene0_id = "ENSG00000147050.14"
gene1_id = "ENSG00000183878.15"

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %% tags=[]
# get percentiles
_tmp = df_plot.loc[(gene0_id, gene1_id), ["ccc", "pearson", "spearman"]]
display(get_percentiles(_tmp))

# %% tags=[]
gene0_id = "ENSG00000115165.9"
gene1_id = "ENSG00000101265.15"

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %% tags=[]
# get percentiles
_tmp = df_plot.loc[(gene0_id, gene1_id), ["ccc", "pearson", "spearman"]]
display(get_percentiles(_tmp))

# %% [markdown] tags=[]
# #### Second selection
#
# Second (after first revision) gene pair selection, which considers both coefficient values and p-value.

# %% tags=[]
gene0_id = "ENSG00000147050.14"
gene1_id = "ENSG00000067048.16"

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %% tags=[]
# get percentiles
_tmp = df_plot.loc[(gene0_id, gene1_id), ["ccc", "pearson", "spearman"]]
display(get_percentiles(_tmp))

# %% [markdown] tags=[]
# ## CCC vs Spearman/Pearson

# %% tags=[]
_coef_fdr_col_name = "spearman_fdr"

# %% tags=[]
_tmp_df = get_gene_pairs(
    "ccc",
    {
        "Clustermatch (high)",
        "Spearman (low)",
        "Pearson (low)",
    },
)

save_gene_pairs(_tmp_df, "clustermatch_vs_pearson_spearman")

display(_tmp_df.shape)
display(_tmp_df)

# %% [markdown] tags=[]
# ### Preview sorting by top coefficient value only

# %% tags=[]
for i in range(min(_tmp_df.shape[0], 10)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ### Preview sorting by top coefficient value and non-significant other p-value

# %% tags=[]
# take a look considering the least significant p-values of the other coefficient
_tmp_df_pval = _tmp_df[
    (_tmp_df[_coef_fdr_col_name] > 1e-5) & (_tmp_df["pearson_fdr"] > 0.05)
]  # .sort_values("spearman_fdr", ascending=False)
display(_tmp_df_pval)

# %% tags=[]
for i in range(min(_tmp_df_pval.shape[0], 10)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df_pval, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ### Preview sorting non-significant other p-value

# %% tags=[]
# take a look considering the least significant p-values of the other coefficient
_tmp_df_other_pval = _tmp_df[(_tmp_df["pearson_fdr"] > 0.05)].sort_values(
    _coef_fdr_col_name, ascending=False
)
display(_tmp_df_other_pval)

# %% tags=[]
for i in range(min(_tmp_df_other_pval.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df_other_pval, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ### Selection

# %% tags=[]
gene_pair_subset = "c_vs_r_rs"

# %% [markdown] tags=[]
# #### Initial selection
#
# Initial (first manuscript submission) gene pair selection considering coefficient values only (but not their p-value):

# %% tags=[]
gene0_id = "ENSG00000162413.16"
gene1_id = "ENSG00000235027.1"

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %% tags=[]
# get percentiles
_tmp = df_plot.loc[(gene0_id, gene1_id), ["ccc", "pearson", "spearman"]]
display(get_percentiles(_tmp))

# %% [markdown] tags=[]
# #### Second selection
#
# Second (after first revision) gene pair selection, which considers both coefficient values and p-value.

# %% tags=[]
# gene0_id = "xxx"
# gene1_id = "yyy"

# plot_and_save_gene_pair(
#     gene_expr_df.T,
#     gene0_id,
#     gene1_id,
#     output_file_subset=gene_pair_subset,
# )

# %% tags=[]
# # get percentiles
# _tmp = df_plot.loc[(gene0_id, gene1_id), ["ccc", "pearson", "spearman"]]
# display(get_percentiles(_tmp))

# %% [markdown] tags=[]
# ## Pearson vs CCC

# %% tags=[]
_coef_fdr_col_name = "ccc_fdr"

# %% tags=[]
_tmp_df = get_gene_pairs(
    "pearson",
    {
        "Clustermatch (low)",
        "Pearson (high)",
    },
)

save_gene_pairs(_tmp_df, "pearson_vs_clustermatch")

display(_tmp_df.shape)
display(_tmp_df)

# %% [markdown] tags=[]
# ### Preview sorting by top coefficient value only

# %% tags=[]
for i in range(min(_tmp_df.shape[0], 10)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ### Preview sorting by top coefficient value and non-significant other p-value

# %% tags=[]
# take a look considering the least significant p-values of the other coefficient
_tmp_df_pval = _tmp_df[
    (_tmp_df[_coef_fdr_col_name] > 0.001)
]  # .sort_values("spearman_fdr", ascending=False)
display(_tmp_df_pval)

# %% tags=[]
for i in range(min(_tmp_df_pval.shape[0], 10)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df_pval, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ### Preview sorting non-significant other p-value

# %% tags=[]
# take a look considering the least significant p-values of the other coefficient
_tmp_df_other_pval = _tmp_df[(_tmp_df[_coef_fdr_col_name] > 0.001)].sort_values(
    _coef_fdr_col_name, ascending=False
)
display(_tmp_df_other_pval)

# %% tags=[]
for i in range(min(_tmp_df_other_pval.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df_other_pval, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ### Selection

# %% tags=[]
gene_pair_subset = "r_vs_c"

# %% [markdown] tags=[]
# #### Initial selection
#
# Initial (first manuscript submission) gene pair selection considering coefficient values only (but not their p-value):

# %% tags=[]
gene0_id = "ENSG00000130598.15"
gene1_id = "ENSG00000177791.11"

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %% tags=[]
# get percentiles
_tmp = df_plot.loc[(gene0_id, gene1_id), ["ccc", "pearson", "spearman"]]
display(get_percentiles(_tmp))

# %% [markdown] tags=[]
# #### Second selection
#
# Second (after first revision) gene pair selection, which considers both coefficient values and p-value.

# %% tags=[]
# gene0_id = "xxx"
# gene1_id = "yyy"

# plot_and_save_gene_pair(
#     gene_expr_df.T,
#     gene0_id,
#     gene1_id,
#     output_file_subset=gene_pair_subset,
# )

# %% tags=[]
# # get percentiles
# _tmp = df_plot.loc[(gene0_id, gene1_id), ["ccc", "pearson", "spearman"]]
# display(get_percentiles(_tmp))

# %% [markdown] tags=[]
# ## Pearson vs Spearman

# %% [markdown] tags=[]
# Listing gene pairs, but not interested in this combination.

# %% tags=[]
_tmp_df = get_gene_pairs(
    "pearson",
    {
        "Spearman (low)",
        "Pearson (high)",
    },
)

display(_tmp_df.shape)
display(_tmp_df)

# %% [markdown] tags=[]
# ### Preview

# %% tags=[]
for i in range(min(_tmp_df.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ### Selection

# %% [markdown] tags=[]
# ## Pearson vs Spearman/CCC

# %% tags=[]
_coef_fdr_col_name = "ccc_fdr"

# %% tags=[]
_tmp_df = get_gene_pairs(
    "pearson",
    {
        "Clustermatch (low)",
        "Spearman (low)",
        "Pearson (high)",
    },
)

save_gene_pairs(_tmp_df, "pearson_vs_clustermatch_spearman")

display(_tmp_df.shape)
display(_tmp_df)

# %% [markdown] tags=[]
# ### Preview sorting by top coefficient value only

# %% tags=[]
for i in range(min(_tmp_df.shape[0], 10)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ### Preview sorting by top coefficient value and non-significant other p-value

# %% tags=[]
# take a look considering the least significant p-values of the other coefficient
_tmp_df_pval = _tmp_df[
    (_tmp_df[_coef_fdr_col_name] > 0.05) & (_tmp_df["spearman_fdr"] > 0.05)
]  # .sort_values("spearman_fdr", ascending=False)
display(_tmp_df_pval)

# %% tags=[]
for i in range(min(_tmp_df_pval.shape[0], 10)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df_pval, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ### Preview sorting non-significant other p-value

# %% tags=[]
# take a look considering the least significant p-values of the other coefficient
_tmp_df_other_pval = _tmp_df[
    (_tmp_df[_coef_fdr_col_name] > 0.05) & (_tmp_df["spearman_fdr"] > 0.05)
].sort_values(_coef_fdr_col_name, ascending=False)
display(_tmp_df_other_pval)

# %% tags=[]
for i in range(min(_tmp_df_other_pval.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df_other_pval, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ### Selection

# %% tags=[]
gene_pair_subset = "r_vs_c_rs"

# %% [markdown] tags=[]
# #### Initial selection
#
# Initial (first manuscript submission) gene pair selection considering coefficient values only (but not their p-value):

# %% tags=[]
gene0_id = "ENSG00000198467.13"
gene1_id = "ENSG00000068976.13"

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %% tags=[]
# get percentiles
_tmp = df_plot.loc[(gene0_id, gene1_id), ["ccc", "pearson", "spearman"]]
display(get_percentiles(_tmp))

# %% [markdown] tags=[]
# #### Second selection
#
# Second (after first revision) gene pair selection, which considers both coefficient values and p-value.

# %% tags=[]
gene0_id = "ENSG00000167644.11"
gene1_id = "ENSG00000161055.3"

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %% tags=[]
# get percentiles
_tmp = df_plot.loc[(gene0_id, gene1_id), ["ccc", "pearson", "spearman"]]
display(get_percentiles(_tmp))

# %% [markdown] tags=[]
# ## Spearman vs Pearson

# %% [markdown] tags=[]
# Listing gene pairs, but not interested in this combination.

# %% tags=[]
_tmp_df = get_gene_pairs(
    "spearman",
    {
        "Spearman (high)",
        "Pearson (low)",
    },
)

display(_tmp_df.shape)
display(_tmp_df)

# %% [markdown] tags=[]
# ### Preview

# %% tags=[]
for i in range(min(_tmp_df.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ### Selection

# %% tags=[]
