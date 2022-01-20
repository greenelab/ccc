# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# It plot specific gene pairs from different subsets of the UpSet plot (intersections) generated before.
#
# The idea of the notebook is to take a look at the patterns found and not found by different methods.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import pandas as pd
import numpy as np

from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

from clustermatch.plots import jointplot
from clustermatch.coef import cm
from clustermatch import conf

# %% [markdown] tags=[]
# # Settings

# %%
DATASET_CONFIG = conf.RECOUNT2FULL
# GTEX_TISSUE = "whole_blood"
GENE_SEL_STRATEGY = "var_pc_log2"

# %% [markdown] tags=[]
# # Paths

# %%
dataset_name = DATASET_CONFIG["RESULTS_DIR"].name
display(dataset_name)

# %% tags=[]
assert (
    conf.MANUSCRIPT["BASE_DIR"] is not None and conf.MANUSCRIPT["BASE_DIR"].exists()
), "Manuscript dir not set"

# %% tags=[]
OUTPUT_FIGURE_DIR = (
    conf.MANUSCRIPT["FIGURES_DIR"] / "coefs_comp" / dataset_name
)
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_FIGURE_DIR)

# %% tags=[]
INPUT_GENE_EXPR_FILE = DATASET_CONFIG["DATA_DIR"] / "recount2_rpkm.pkl"
display(INPUT_GENE_EXPR_FILE)

assert INPUT_GENE_EXPR_FILE.exists()

# %% tags=[]
INPUT_GENE_PAIRS_INTERSECTIONS_FILE = (
    DATASET_CONFIG["GENE_PAIR_INTERSECTIONS"]
    / f"gene_pair_intersections-{dataset_name}-{GENE_SEL_STRATEGY}.pkl"
)
display(INPUT_GENE_PAIRS_INTERSECTIONS_FILE)

assert INPUT_GENE_PAIRS_INTERSECTIONS_FILE.exists()

# %% [markdown] tags=[]
# # Data

# %% [markdown] tags=[]
# ## Gene Ensembl ID -> Symbol mapping

# %%
gene_map = pd.read_pickle(
    DATASET_CONFIG["DATA_DIR"] / f"recount2_gene_ids_mappings.pkl"
).set_index("ensembl_gene_id").squeeze().sort_index()

# %%
gene_map.shape

# %%
gene_map.head()

# %%
# remove genes with no mappings
gene_map = gene_map[gene_map.apply(len) > 0]

# %%
# remove duplicated entries
gene_map = gene_map.drop_duplicates()

# %%
gene_map.shape

# %%
gene_map.index.is_unique

# %%
gene_map.is_unique

# %%
gene_map["ENSG00000000003"]

# %% [markdown] tags=[]
# ## Gene pairs intersection

# %%
df_plot = pd.read_pickle(INPUT_GENE_PAIRS_INTERSECTIONS_FILE)

# %%
df_plot.shape

# %%
df_plot.head()

# %%
df_plot_all_genes = df_plot.index.get_level_values(1).unique().union(df_plot.index.get_level_values(0).unique())
display(df_plot_all_genes)

# %% [markdown] tags=[]
# ## Gene expression

# %%
gene_expr_df = pd.read_pickle(INPUT_GENE_EXPR_FILE).loc[df_plot_all_genes].sort_index()

# %%
gene_expr_df.shape

# %%
gene_expr_df.head()

# %% [markdown]
# # Look at specific gene pair cases

# %% [markdown]
# ## Prepare data

# %%
# add columns with ranks
df_r_data = pd.concat(
    [
        df_plot,
        df_plot[["clustermatch", "pearson", "spearman"]]
        .rank()
        .rename(
            columns={
                "clustermatch": "clustermatch_rank",
                "pearson": "pearson_rank",
                "spearman": "spearman_rank",
            }
        ),
    ],
    axis=1,
).sort_index()

# %%
df_r_data.head()

# %%
df_r_data_boolean_cols = set(
    [x for x in df_r_data.columns if " (high)" in x or " (low)" in x]
)

# %%
df_r_data_boolean_cols


# %% [markdown]
# ## Functions

# %%
def get_gene_symbol(gene_ensemble_id):
    if gene_ensemble_id not in gene_map.index:
        return f"{gene_ensemble_id} (no symbol)"
    
    gene_symbols = gene_map.loc[[gene_ensemble_id]]
    
    if gene_symbols.shape[0] == 1:
        return gene_symbols.iloc[0]
    
    return " / ".join(gene_symbols)

# testing
assert get_gene_symbol("ENSG00000260894") == "ENSG00000260894 (no symbol)"
assert get_gene_symbol("ENSG00000000003") == "TSPAN6"
assert get_gene_symbol("ENSG00000276085") == "CCL3L3 / CCL3L1"


# %%
def plot_gene_pair(top_pairs_df, idx, bins="log", plot_gene_ids=True): #, downsample: int=None):
    gene0, gene1 = top_pairs_df.iloc[idx].name
    display((gene0, gene1))

    gene0_symbol = get_gene_symbol(gene0)
    gene1_symbol = get_gene_symbol(gene1)
    display((gene0_symbol, gene1_symbol))

    _pearson, _spearman, _clustermatch = top_pairs_df.loc[
        (gene0, gene1), ["pearson", "spearman", "clustermatch"]
    ].tolist()
    
    _gene_expr_df_sample = gene_expr_df.T
#     if downsample is not None:
#         _gene_expr_df_sample = gene_expr_df.sample(downsample, random_state=downsample).T
        
#         # add potentially missing genes to plot
#         if gene0 not in _gene_expr_df_sample.columns:
#             _gene_expr_df_sample[gene0] = gene_expr_df.loc[gene0]
            
#         if gene1 not in _gene_expr_df_sample.columns:
#             _gene_expr_df_sample[gene1] = gene_expr_df.loc[gene1]
    
    p = jointplot(
        data=_gene_expr_df_sample,
        x=gene0,
        y=gene1,
        # kind="hex",
        bins=bins,
        add_corr_coefs=False,
        # rasterized=True,
        # ylim=(0, 500),
    )
    
    # p = sns.jointplot(
    #     data=_gene_expr_df_sample,
    #     x=gene0,
    #     y=gene1,
    #     kind="hex",
    #     bins=bins,
    #     rasterized=True,
    #     # ylim=(0, 500),
    # )
    
    if plot_gene_ids:
        p.ax_joint.set_xlabel(f"{gene0}\n{gene0_symbol}")
        p.ax_joint.set_ylabel(f"{gene1}\n{gene1_symbol}")
    else:
        p.ax_joint.set_xlabel(f"{gene0_symbol}", fontstyle="italic")
        p.ax_joint.set_ylabel(f"{gene1_symbol}", fontstyle="italic")

    _title = f"$c={_clustermatch:.2f}$  $r={_pearson:.2f}$    $r_s={_spearman:.2f}$"
    p.fig.suptitle(_title)

    return p


# %%
def get_gene_pairs(first_coef, query_set):
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


# %%
def plot_and_save_gene_pair(data, gene0_id, gene1_id, output_file_subset):
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

        output_file = OUTPUT_FIGURE_DIR / f"genes-{output_file_subset}-{gene0_symbol}_vs_{gene1_symbol}.svg"
        display(output_file)
        
        plt.savefig(
            output_file,
            bbox_inches="tight",
            dpi=300,
            facecolor="white",
        )


# %% [markdown] tags=[]
# ## Clustermatch/Pearson vs Spearman

# %%
_tmp_df = get_gene_pairs(
    "clustermatch",
    {
        "Clustermatch (high)",
        "Pearson (high)",
        "Spearman (low)",
    },
)

display(_tmp_df.shape)
display(_tmp_df)

# %% [markdown]
# ### Preview

# %%
for i in range(min(_tmp_df.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ### Selection

# %%
gene_pair_subset = "c_r_vs_rs"

gene0_id = ""
gene1_id = ""

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %% [markdown] tags=[]
# ## Clustermatch/Spearman vs Pearson

# %%
_tmp_df = get_gene_pairs(
    "clustermatch",
    {
        "Clustermatch (high)",
        "Spearman (high)",
        "Pearson (low)",
    },
)

display(_tmp_df.shape)
display(_tmp_df)

# %% [markdown]
# ### Preview

# %%
for i in range(min(_tmp_df.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ### Selection

# %%
gene_pair_subset = "c_rs_vs_r"

gene0_id = ""
gene1_id = ""

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %%
IT WOULD BE GOOD, WHEN PICKING PEARSON HIGH, TO SORT BY PEARSON, BECAUSE ITS VALUES ARE GENERALLY VERY LOW

# %% [markdown] tags=[]
# ## Clustermatch vs Pearson

# %%
_tmp_df = get_gene_pairs(
    "clustermatch",
    {
        "Clustermatch (high)",
        "Pearson (low)",
    },
)

display(_tmp_df.shape)
display(_tmp_df)

# %% [markdown]
# ### Preview

# %%
for i in range(min(_tmp_df.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown]
# ### Explore

# %%
from scipy.stats.contingency import expected_freq

# %%
from clustermatch.sklearn.metrics import get_contingency_matrix


# %% tags=[]
def get_cm_contingency_table(max_parts, parts):
    """
    TODO
    """
    # get the clustermatch partitions that maximize the coefficient
    x_max_part = parts[0][max_parts[0]]
    
    y_max_part = parts[1][max_parts[1]]
    new_y_max_part = np.full(y_max_part.shape, np.nan)
    for new_k, k in enumerate(np.flip(np.unique(y_max_part))):
        new_y_max_part[y_max_part == k] = new_k
    
    return get_contingency_matrix(new_y_max_part, x_max_part)


# %%
def get_adjusted_contingency_matrix(cont_mat):
    n = int(cont_mat.sum())
    nis = np.sum(cont_mat, axis=1)
    njs = np.sum(cont_mat, axis=0)

    adj_cont_mat = np.full(cont_mat.shape, np.nan)
    for i in range(cont_mat.shape[0]):
        ni = int(nis[i])

        for j in range(cont_mat.shape[1]):
            nj = int(njs[j])

            nij = int(cont_mat[i, j])

            adj_cont_mat[i, j] = (nij * (nij - 1)) / ((ni * (ni - 1) * nj * (nj - 1)) / (n * (n - 1)))
    
    return adj_cont_mat


# %%
gene0, gene1 = ('ENSG00000202293', 'ENSG00000199077')

gene0_data = gene_expr_df.loc[gene0]
gene1_data = gene_expr_df.loc[gene1]

# gene0_data = gene0_data[gene0_data < 37]
# gene1_data = gene1_data[gene1_data < 41]

_gene_expr_df_sample = pd.concat([gene0_data, gene1_data], axis=1)
_gene_expr_df_sample = _gene_expr_df_sample.dropna()
display(_gene_expr_df_sample.shape)

p = jointplot(
    data=_gene_expr_df_sample,
    x=gene0,
    y=gene1,
    bins=None,
    add_corr_coefs=False,
)

# %%
sns.displot(
    data=_gene_expr_df_sample,
    x=gene0,
    y=gene1,
    kind="hist",
    bins=10,
)

# %%
sns.displot(
    data=_gene_expr_df_sample,
    x=gene0,
    y=gene1,
    kind="kde",
    # thresh=0.75,
)

# %%
_gene_expr_df_sample.describe()

# %%
_gene_expr_df_sample.quantile(0.95)

# %%
x = _gene_expr_df_sample.iloc[:, 0]
y = _gene_expr_df_sample.iloc[:, 1]
c, max_parts, parts = cm(x, y, return_parts=True, internal_n_clusters=[10])
display(c)

# %%
cont_mat = get_cm_contingency_table(max_parts, parts)

# %%
cont_mat

# %%
_cont_mat_expected = expected_freq(cont_mat)
sns.heatmap(cont_mat / _cont_mat_expected, annot=True, fmt='.2f', cmap="Blues")

# %%
adj_cont_mat = get_adjusted_contingency_matrix(cont_mat)

# %%
sns.heatmap(adj_cont_mat, annot=True, fmt='.1f', cmap="Blues")

# %%

# %%
import statsmodels.formula.api as smf
from sklearn.preprocessing import scale

# %%
x = _gene_expr_df_sample.iloc[:, 0]
y = _gene_expr_df_sample.iloc[:, 1]

# %%
x_s = scale(x)

# %%
z = np.polyfit(x_s, y, 4)

# %%
m = np.poly1d(z)

# %%
m

# %%
df = _gene_expr_df_sample.rename(columns={gene0: "x", gene1: "y"})
df["x"] = scale(df["x"])
results = smf.ols(formula='y ~ m(x)', data=df).fit()

# %%
results.summary()

# %%
df = df.sort_values("x")
x = df["x"]
y = df["y"]

fig, ax = plt.subplots()
ax.plot(x, y, "o", label='data')
# ax.plot(x, m(scale(x)), "--", label='fit')
ax.legend()
ax.set_xlim(0, 50)
ax.set_ylim(-10, 200)

# %% [markdown]
# ### Selection

# %%
gene_pair_subset = "c_vs_r"

gene0_id = ""
gene1_id = ""

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %% [markdown] tags=[]
# ## Clustermatch vs Spearman

# %%
_tmp_df = get_gene_pairs(
    "clustermatch",
    {
        "Clustermatch (high)",
        "Spearman (low)",
    },
)

display(_tmp_df.shape)
display(_tmp_df)

# %% [markdown]
# ### Preview

# %%
for i in range(min(_tmp_df.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown]
# ### Explore

# %%
gene0, gene1 = ('ENSG00000221533', 'ENSG00000206630')

gene0_data = gene_expr_df.loc[gene0]
gene1_data = gene_expr_df.loc[gene1]

# gene0_data = gene0_data[gene0_data < 37]
# gene1_data = gene1_data[gene1_data < 41]

_gene_expr_df_sample = pd.concat([gene0_data, gene1_data], axis=1)
_gene_expr_df_sample = _gene_expr_df_sample.dropna()
display(_gene_expr_df_sample.shape)

p = jointplot(
    data=_gene_expr_df_sample,
    x=gene0,
    y=gene1,
    bins=None,
    add_corr_coefs=False,
)

# %%
sns.displot(
    data=_gene_expr_df_sample,
    x=gene0,
    y=gene1,
    kind="hist",
    bins=10,
)

# %%
sns.displot(
    data=_gene_expr_df_sample,
    x=gene0,
    y=gene1,
    kind="kde",
    # thresh=0.01,
)

# %%
_gene_expr_df_sample.describe()

# %%
_gene_expr_df_sample.quantile(0.90)

# %%
x = _gene_expr_df_sample.iloc[:, 0]
y = _gene_expr_df_sample.iloc[:, 1]
c, max_parts, parts = cm(x, y, return_parts=True)
display(c)

# %%
cont_mat = get_cm_contingency_table(max_parts, parts)

# %%
cont_mat

# %%
_cont_mat_expected = expected_freq(cont_mat)
sns.heatmap(cont_mat / _cont_mat_expected, annot=True, fmt='.2f', cmap="Blues")

# %%
adj_cont_mat = get_adjusted_contingency_matrix(cont_mat)

# %%
sns.heatmap(adj_cont_mat, annot=True, fmt='.1f', cmap="Blues")

# %% [markdown]
# ### Selection

# %%
gene_pair_subset = "c_vs_rs"

gene0_id = ""
gene1_id = ""

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %% [markdown] tags=[]
# ## Clustermatch vs Spearman/Pearson

# %%
_tmp_df = get_gene_pairs(
    "clustermatch",
    {
        "Clustermatch (high)",
        "Spearman (low)",
        "Pearson (low)",
    },
)

display(_tmp_df.shape)
display(_tmp_df)

# %% [markdown]
# ### Preview

# %%
for i in range(min(_tmp_df.shape[0], 10)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown]
# ### Explore

# %%
gene0, gene1 = ('ENSG00000260894', 'ENSG00000277370')

gene0_data = gene_expr_df.loc[gene0]
gene1_data = gene_expr_df.loc[gene1]

# gene0_data = gene0_data[gene0_data < 37]
# gene1_data = gene1_data[gene1_data < 41]

_gene_expr_df_sample = pd.concat([gene0_data, gene1_data], axis=1)
_gene_expr_df_sample = _gene_expr_df_sample.dropna()
display(_gene_expr_df_sample.shape)

p = jointplot(
    data=_gene_expr_df_sample,
    x=gene0,
    y=gene1,
    bins=None,
    add_corr_coefs=False,
)

# %%
sns.displot(
    data=_gene_expr_df_sample,
    x=gene0,
    y=gene1,
    kind="hist",
    bins=10,
)

# %%
sns.displot(
    data=_gene_expr_df_sample,
    x=gene0,
    y=gene1,
    kind="kde",
    thresh=0.75,
    # levels=10,
    # kind="hist",
    # bins=10,
)

# %%
_gene_expr_df_sample.describe()

# %%
_gene_expr_df_sample.quantile(0.90)

# %%
x = _gene_expr_df_sample.iloc[:, 0]
y = _gene_expr_df_sample.iloc[:, 1]
c, max_parts, parts = cm(x, y, return_parts=True)
display(c)

# %%
cont_mat = get_cm_contingency_table(max_parts, parts)

# %%
cont_mat

# %%
_cont_mat_expected = expected_freq(cont_mat)
sns.heatmap(cont_mat / _cont_mat_expected, annot=True, fmt='.2f', cmap="Blues")

# %%
adj_cont_mat = get_adjusted_contingency_matrix(cont_mat)

# %%
sns.heatmap(adj_cont_mat, annot=True, fmt='.1f', cmap="Blues")

# %% [markdown]
# ### Selection

# %%
gene_pair_subset = "c_vs_r_rs"

gene0_id = ""
gene1_id = ""

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %% [markdown] tags=[]
# ## Pearson vs Clustermatch

# %%
_tmp_df = get_gene_pairs(
    "pearson",
    {
        "Clustermatch (low)",
        "Pearson (high)",
    },
)

display(_tmp_df.shape)
display(_tmp_df)

# %% [markdown]
# ### Preview

# %%
for i in range(min(_tmp_df.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown]
# ### Selection

# %%
gene_pair_subset = "r_vs_c"

gene0_id = ""
gene1_id = ""

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %% [markdown] tags=[]
# ## Pearson vs Spearman

# %%
_tmp_df = get_gene_pairs(
    "pearson",
    {
        "Spearman (low)",
        "Pearson (high)",
    },
)

display(_tmp_df.shape)
display(_tmp_df)

# %% [markdown]
# ### Preview

# %%
for i in range(min(_tmp_df.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown]
# ### Selection

# %% [markdown] tags=[]
# ## Pearson vs Spearman/Clustermatch

# %%
_tmp_df = get_gene_pairs(
    "pearson",
    {
        "Clustermatch (low)",
        "Spearman (low)",
        "Pearson (high)",
    },
)

display(_tmp_df.shape)
display(_tmp_df)

# %% [markdown]
# ### Preview

# %%
for i in range(min(_tmp_df.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown]
# ### Selection

# %%
gene_pair_subset = "r_vs_c_rs"

gene0_id = ""
gene1_id = ""

plot_and_save_gene_pair(
    gene_expr_df.T,
    gene0_id,
    gene1_id,
    output_file_subset=gene_pair_subset,
)

# %% [markdown] tags=[]
# ## Spearman vs Clustermatch

# %%
_tmp_df = get_gene_pairs(
    "spearman",
    {
        "Spearman (high)",
        "Clustermatch (low)",
    },
)

display(_tmp_df.shape)
display(_tmp_df)

# %% [markdown]
# ### Preview

# %%
for i in range(min(_tmp_df.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown]
# ### Explore

# %%
gene0, gene1 = ('ENSG00000213442', 'ENSG00000199153')

gene0_data = gene_expr_df.loc[gene0]
gene1_data = gene_expr_df.loc[gene1]

# gene0_data = gene0_data[gene0_data < 500]
# gene1_data = gene1_data[gene1_data < 100]

_gene_expr_df_sample = pd.concat([gene0_data, gene1_data], axis=1)
_gene_expr_df_sample = _gene_expr_df_sample.dropna()
display(_gene_expr_df_sample.shape)

p = jointplot(
    data=_gene_expr_df_sample,
    x=gene0,
    y=gene1,
    bins=None,
    add_corr_coefs=False,
)

# %%
sns.displot(
    data=_gene_expr_df_sample,
    x=gene0,
    y=gene1,
    # kind="kde",
    # thresh=0.50,
    kind="hist",
    bins=10,
)

# %%
_gene_expr_df_sample.describe().applymap(str)

# %%
x = _gene_expr_df_sample.iloc[:, 0]
y = _gene_expr_df_sample.iloc[:, 1]
c, max_parts, parts = cm(x, y, return_parts=True)
display(c)

# %%
pd.Series(parts[1][0]).value_counts()

# %%
cont_mat = get_cm_contingency_table(max_parts, parts)

# %%
cont_mat

# %%
sns.heatmap(cont_mat / cont_mat.sum(), annot=True, fmt='.2f', cmap="Blues")

# %%
adj_cont_mat = get_adjusted_contingency_matrix(cont_mat)

# %%
sns.heatmap(adj_cont_mat, annot=True, fmt='.1f', cmap="Blues")

# %% [markdown] tags=[]
# ## Spearman vs Pearson

# %%
_tmp_df = get_gene_pairs(
    "spearman",
    {
        "Spearman (high)",
        "Pearson (low)",
    },
)

display(_tmp_df.shape)
display(_tmp_df)

# %% [markdown]
# ### Preview

# %%
for i in range(min(_tmp_df.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown]
# ### Selection

# %%
