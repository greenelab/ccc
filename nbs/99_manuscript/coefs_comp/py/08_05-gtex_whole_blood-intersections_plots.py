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

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import pandas as pd
import numpy as np

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
# ## Gene pairs percentiles

# %% tags=[]
df_plot_percentiles = df_plot[["ccc", "pearson", "spearman"]].quantile(
    np.arange(0.1, 1.0, 0.01)
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

    _pearson, _spearman, _clustermatch = top_pairs_df.loc[
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

    _title = f"$c={_clustermatch:.2f}$  $p={_pearson:.2f}$    $s={_spearman:.2f}$"
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


# %% [markdown] tags=[]
# ## CCC/Spearman vs Pearson

# %% tags=[]
_tmp_df = get_gene_pairs(
    "ccc",
    {
        "Clustermatch (high)",
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
gene_pair_subset = "c_rs_vs_r"

gene0_id = "ENSG00000135094.10"
gene1_id = "ENSG00000111537.4"

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %% tags=[]
gene_pair_subset = "c_rs_vs_r"

gene0_id = "ENSG00000130208.9"
gene1_id = "ENSG00000177606.6"

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %% [markdown] tags=[]
# ## CCC vs Pearson

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
gene_pair_subset = "c_vs_r"

gene0_id = "ENSG00000275385.1"
gene1_id = "ENSG00000160446.18"

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %% [markdown] tags=[]
# ## CCC vs Spearman

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
# ### Preview

# %% tags=[]
for i in range(min(_tmp_df.shape[0], 30)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ### Selection

# %% tags=[]
gene_pair_subset = "c_vs_rs"

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
df_plot.loc[(gene0_id, gene1_id), ["ccc", "pearson", "spearman"]]

# %% tags=[]
gene_pair_subset = "c_vs_rs"

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
df_plot.loc[(gene0_id, gene1_id), ["ccc", "pearson", "spearman"]]

# %% [markdown] tags=[]
# ## CCC vs Spearman/Pearson

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
# ### Preview

# %% tags=[]
for i in range(min(_tmp_df.shape[0], 10)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ### Selection

# %% tags=[]
gene_pair_subset = "c_vs_r_rs"

gene0_id = "ENSG00000162413.16"
gene1_id = "ENSG00000235027.1"

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %% [markdown] tags=[]
# ## Pearson vs CCC

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
gene_pair_subset = "r_vs_c"

gene0_id = "ENSG00000130598.15"
gene1_id = "ENSG00000177791.11"

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %% [markdown] tags=[]
# ## Pearson vs Spearman

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
gene_pair_subset = "r_vs_c_rs"

gene0_id = "ENSG00000198467.13"
gene1_id = "ENSG00000068976.13"

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %% [markdown] tags=[]
# ## Spearman vs Pearson

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
