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
# It analyzes different strategies to take the genes from GTEx data with the highest variability, being this variability measured with different strategies: variance (`var`), coefficient of variation (`cv`) and mean absolute variation (`mad`) applied on two different versions of the data: 1) the raw TPM-normalized gene expression data (here refered to as `raw`), and 2) the log2-transformed version of the raw data (here refered to as `log2` and `pc_log2`).

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from tqdm import tqdm

from clustermatch import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
N_TOP_GENES_MAX_VARIANCE = 5000

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
INPUT_DIR = conf.GTEX["DATA_DIR"] / "data_by_tissue"
display(INPUT_DIR)

# %% tags=[]
OUTPUT_DIR = conf.GTEX["GENE_SELECTION_DIR"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %% [markdown] tags=[]
# # Functions

# %% tags=[]
PCA_OPTIONS = {
    "n_components": 5,
    "random_state": 0,
}


# %% tags=[]
def standardize(data):
    return pd.DataFrame(
        data=scale(data),
        index=data.index.copy(),
        columns=data.columns.copy(),
    )


# %% tags=[]
def plot_pca(data, std=True):
    if std:
        data = standardize(data)

    clf = PCA(**PCA_OPTIONS)
    pca_data = clf.fit_transform(data)

    pca_data = pd.DataFrame(
        data=pca_data,
        index=data.index.copy(),
        columns=[f"PCA{i+1}" for i in range(pca_data.shape[1])],
    )

    g = sns.pairplot(data=pca_data)
    display(g)


# %% [markdown] tags=[]
# # Compare different criteria to select highly variable genes

# %% tags=[]
# I will store here the top genes selected by each method
top_genes_var = {}

# %% [markdown] tags=[]
# ## Get test data

# %% tags=[]
test_data = pd.read_pickle(INPUT_DIR / "gtex_v8_data_whole_blood.pkl")

# %% tags=[]
test_data.shape

# %% tags=[]
test_data.head()

# %% tags=[]
test_data_desc = pd.Series(test_data.to_numpy().flatten()).describe()
display(test_data_desc)

assert test_data_desc["min"] == 0.0
assert test_data_desc["max"] < 7.5e5

# %% [markdown] tags=[]
# ## Get test data in log2

# %% [markdown] tags=[]
# Here I attempt a direct log transformation without any change to the raw data, and replacing -inf values by the minimum.

# %% tags=[]
log2_test_data = np.log2(test_data)

# %% tags=[]
log2_test_data.shape

# %% tags=[]
log2_test_data.head()

# %% tags=[]
# get minimum values by removing -np.inf first
sample_min_values = (
    pd.Series(log2_test_data.replace(-np.inf, np.nan).to_numpy().flatten())
    .dropna()
    .sort_values()
)

# %% tags=[]
sample_min_values.head()

# %% tags=[]
# get the min value and replace -np.inf by it
log2_min_value = sample_min = sample_min_values.iloc[0]
display(log2_min_value)
assert log2_min_value < -13.0
assert log2_min_value > -13.5

# %% tags=[]
log2_test_data = log2_test_data.replace(-np.inf, log2_min_value * 1.3)

# %% tags=[]
log2_test_data.shape

# %% tags=[]
assert (
    log2_test_data.iloc[:, [0]].squeeze().loc["ENSG00000278267.1"].round(5) == -17.28173
)

# %% tags=[]
log2_test_data.head()

# %% tags=[]
log2_test_data.iloc[:10, :].T.describe()

# %% tags=[]
# get some stats
log2_test_data_desc = pd.Series(log2_test_data.to_numpy().flatten()).describe()
display(log2_test_data_desc)

# %% [markdown] tags=[]
# ## Get test data in log2 after pseudocount

# %% [markdown] tags=[]
# Here I try another approach to log-transform the data by using pseudocounts. See:
#  - https://github.com/greenelab/clustermatch-gene-expr/pull/4#discussion_r698793383

# %% tags=[]
log2_pc_test_data = np.log2(test_data + 1)

# %% tags=[]
log2_pc_test_data.shape

# %% tags=[]
# get some stats
log2_pc_test_data_desc = pd.Series(log2_pc_test_data.to_numpy().flatten()).describe()
display(log2_pc_test_data_desc)

# %% tags=[]
log2_pc_test_data.head()

# %% [markdown] tags=[]
# ## On TPM-normalized data (raw)

# %% [markdown] tags=[]
# ### Variance

# %% tags=[]
exp_id = "var_raw"

# %% tags=[]
top_genes_var[exp_id] = (
    test_data.var(axis=1).sort_values(ascending=False).head(N_TOP_GENES_MAX_VARIANCE)
)

# %% tags=[]
top_genes_var[exp_id]

# %% [markdown] tags=[]
# Here I take a quick look at how these selected genes look like in a PCA plot. I do not use PCA to select genes, just a visual assessment.

# %% tags=[]
selected_data = test_data.loc[top_genes_var[exp_id].index]

# %% tags=[]
selected_data.shape

# %% tags=[]
plot_pca(selected_data)

# %% [markdown] tags=[]
# ### Coefficient of variation

# %% tags=[]
exp_id = "cv_raw"

# %% tags=[]
top_genes_var[exp_id] = (
    (test_data.std(axis=1) / test_data.mean(axis=1))
    .sort_values(ascending=False)
    .head(N_TOP_GENES_MAX_VARIANCE)
)

# %% tags=[]
top_genes_var[exp_id]

# %% [markdown] tags=[]
# ### Mean absolute variation

# %% tags=[]
exp_id = "mad_raw"

# %% tags=[]
top_genes_var[exp_id] = (
    test_data.mad(axis=1).sort_values(ascending=False).head(N_TOP_GENES_MAX_VARIANCE)
)

# %% tags=[]
top_genes_var[exp_id]

# %% [markdown] tags=[]
# ## On log2 TPM-normalized data

# %% [markdown] tags=[]
# ### Variance

# %% tags=[]
exp_id = "var_log2"

# %% tags=[]
top_genes_var[exp_id] = (
    log2_test_data.var(axis=1)
    .sort_values(ascending=False)
    .head(N_TOP_GENES_MAX_VARIANCE)
)

# %% tags=[]
top_genes_var[exp_id]

# %% [markdown] tags=[]
# Here I take a quick look at how these selected genes look like in a PCA plot. I do not use PCA to select genes, just a visual assessment.

# %% tags=[]
# plot on raw
selected_data = test_data.loc[top_genes_var[exp_id].index]

# %% tags=[]
selected_data.shape

# %% tags=[]
plot_pca(selected_data)

# %% tags=[]
# plot on log2
selected_data = log2_test_data.loc[top_genes_var[exp_id].index]

# %% tags=[]
selected_data.shape

# %% tags=[]
plot_pca(selected_data)

# %% [markdown] tags=[]
# ### Coefficient of variation

# %% tags=[]
exp_id = "cv_log2"

# %% tags=[]
top_genes_var[exp_id] = (
    (log2_test_data.std(axis=1) / log2_test_data.mean(axis=1))
    .sort_values(ascending=False)
    .head(N_TOP_GENES_MAX_VARIANCE)
)

# %% tags=[]
top_genes_var[exp_id]

# %% [markdown] tags=[]
# ### Mean absolute variation

# %% tags=[]
exp_id = "mad_log2"

# %% tags=[]
top_genes_var[exp_id] = (
    log2_test_data.mad(axis=1)
    .sort_values(ascending=False)
    .head(N_TOP_GENES_MAX_VARIANCE)
)

# %% tags=[]
top_genes_var[exp_id]

# %% [markdown] tags=[]
# ## On pseudocount/log2 TPM-normalized data

# %% [markdown] tags=[]
# ### Variance

# %% tags=[]
exp_id = "var_pc_log2"

# %% tags=[]
top_genes_var[exp_id] = (
    log2_pc_test_data.var(axis=1)
    .sort_values(ascending=False)
    .head(N_TOP_GENES_MAX_VARIANCE)
)

# %% tags=[]
top_genes_var[exp_id]

# %% [markdown] tags=[]
# Here I take a quick look at how these selected genes look like in a PCA plot. I do not use PCA to select genes, just a visual assessment.

# %% tags=[]
# plot on raw
selected_data = test_data.loc[top_genes_var[exp_id].index]

# %% tags=[]
selected_data.shape

# %% tags=[]
plot_pca(selected_data)

# %% tags=[]
# plot on pc_log2
selected_data = log2_pc_test_data.loc[top_genes_var[exp_id].index]

# %% tags=[]
selected_data.shape

# %% tags=[]
plot_pca(selected_data)

# %% [markdown] tags=[]
# ### Coefficient of variation

# %% tags=[]
exp_id = "cv_pc_log2"

# %% tags=[]
top_genes_var[exp_id] = (
    (log2_pc_test_data.std(axis=1) / log2_pc_test_data.mean(axis=1))
    .sort_values(ascending=False)
    .head(N_TOP_GENES_MAX_VARIANCE)
)

# %% tags=[]
top_genes_var[exp_id]

# %% [markdown] tags=[]
# ### Mean absolute deviation

# %% tags=[]
exp_id = "mad_pc_log2"

# %% tags=[]
top_genes_var[exp_id] = (
    log2_pc_test_data.mad(axis=1)
    .sort_values(ascending=False)
    .head(N_TOP_GENES_MAX_VARIANCE)
)

# %% tags=[]
top_genes_var[exp_id]


# %% [markdown] tags=[]
# ## Do selected genes with different methods overlap?

# %% tags=[]
def overlap(x, y):
    ov = set(x).intersection(set(y))
    return len(ov)


# %% tags=[]
assert overlap([1, 2, 3], [4, 5, 6]) == 0
assert overlap([1, 2, 3], [2, 3, 4]) == 2

# %% tags=[]
genes_selection_methods = list(top_genes_var.keys())

display(genes_selection_methods)
assert len(genes_selection_methods) == 9

# %% tags=[]
_gene_sets = np.array(
    [top_genes_var[x].index.tolist() for x in genes_selection_methods]
)

# %% tags=[]
_gene_sets[:2]

# %% tags=[]
assert overlap(_gene_sets[0], _gene_sets[0]) == 5000

# %% tags=[]
_tmp = squareform(pdist(_gene_sets, metric=overlap))
np.fill_diagonal(_tmp, _gene_sets[0].shape[0])
_tmp = pd.DataFrame(
    _tmp, index=genes_selection_methods, columns=genes_selection_methods
)

display(_tmp)

# %% [markdown] tags=[]
# Some methods select very different sets of highly variable genes. `cv_*` methods do not seem to agree much with the rest.
#
# `var_*` and `mad_*` are similar, so it is expected their large overlap among the same data version. However, these two approaches (`var` and `mad`) also agree quite a lot between data versions `raw` and `pc_log2`.

# %% tags=[]
# get list of methods that agree more with the rest
_tmp_top = (_tmp.sum() - 5000).sort_values(ascending=False)
display(_tmp_top)

assert _tmp_top.index[:4].tolist() == [
    "var_pc_log2",
    "mad_pc_log2",
    "var_raw",
    "mad_raw",
]

# %% [markdown] tags=[]
# # How different are genes selected by `raw` and `pc_log2`?

# %% [markdown] tags=[]
# Here I try to see how different are the expression distribution of genes selected using `raw` and `pc_log2`.

# %% tags=[]
# get the rank for each method for easier comparison
genes_df = pd.DataFrame(top_genes_var).rank()

# %% tags=[]
genes_df.shape

# %% tags=[]
genes_df.head()

# %% tags=[]
_tmp = genes_df.describe()
display(_tmp)

assert (_tmp.loc["min"] == 1.0).all()
assert (_tmp.loc["max"] == 5000.0).all()

# %% tags=[]
cols = ["var_raw", "var_pc_log2"]


# %% tags=[]
def plot_genes_kde(_gene_ids):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    axs = axs.flatten()

    # plot density on raw
    ax = axs[0]
    for gene_id in _gene_ids:
        a = sns.kdeplot(data=test_data.T, x=gene_id, ax=ax, label=gene_id)
        a.set_xlabel(None)
    ax.set_title("raw")

    # same genes, but plot density on pc_log2
    ax = axs[1]
    for gene_id in _gene_ids:
        a = sns.kdeplot(data=log2_pc_test_data.T, x=gene_id, ax=ax, label=gene_id)
        a.set_xlabel(None)
    ax.set_title("pseudocount (pc) log2")
    ax.legend()


# %% [markdown] tags=[]
# ## Genes selected in both raw and pc_log2

# %% tags=[]
genes_df.loc[
    list(set(top_genes_var["var_raw"].index) & set(top_genes_var["var_pc_log2"].index)),
    cols,
].sort_values("var_raw", ascending=False).head()

# %% tags=[]
_gene_ids = [
    "ENSG00000163220.10",  # large in both
    "ENSG00000188536.12",  # smaller in log2
]

# %% tags=[]
plot_genes_kde(_gene_ids)

# %% [markdown] tags=[]
# ## Genes selected in raw only

# %% tags=[]
_tmp = genes_df.loc[
    set(top_genes_var["var_raw"].index) - set(top_genes_var["var_pc_log2"].index), cols
].sort_values("var_raw", ascending=False)

display(_tmp.shape)
display(_tmp.head())

# %% tags=[]
_gene_ids = [
    "ENSG00000087086.14",
    "ENSG00000234745.10",
    "ENSG00000198712.1",
]

# %% tags=[]
plot_genes_kde(_gene_ids)

# %% [markdown] tags=[]
# These are the top three genes selected by `var_raw` and not by `var_pc_log2`. Distributions seem similar with different means.

# %% [markdown] tags=[]
# ## Genes selected in pc_log2 only

# %% tags=[]
_tmp = genes_df.loc[
    set(top_genes_var["var_pc_log2"].index) - set(top_genes_var["var_raw"].index), cols
].sort_values("var_pc_log2", ascending=False)

display(_tmp.shape)
display(_tmp.head())

# %% tags=[]
_gene_ids = [
    "ENSG00000241112.1",
    "ENSG00000198692.9",
    "ENSG00000254288.1",
]

# %% tags=[]
plot_genes_kde(_gene_ids)

# %% [markdown] tags=[]
# **CONCLUSION:** Both `var_raw` (that is, the strategy that selects the top genes with highest variance on raw TPM-normalized data) and `var_pc_log2` (highest variance on pseudocount log2-transformed TPM-normalized data) agree on most genes. The difference seem to be that `pc_log2` is more sensitive to genes that are mostly not-expressed and expressed only on some conditions (bimodal distribution in the pc_log2 plot on the right), which might capture important genes such as transcriptor factors (see https://www.biorxiv.org/content/10.1101/2020.02.13.944777v1).

# %% [markdown] tags=[]
# # Is correlation affected by a log-transformation?

# %% tags=[]
_genes_intersect = sorted(
    list(set(top_genes_var["var_pc_log2"].index) & set(top_genes_var["var_raw"].index))
)
display(len(_genes_intersect))
display(_genes_intersect[:5])

assert len(_genes_intersect) == 3330

# %% [markdown] tags=[]
# ## Pearson

# %% tags=[]
raw_corr = test_data.loc[_genes_intersect[:10]].T.corr(method="pearson")
display(raw_corr.shape)
display(raw_corr)

assert raw_corr.shape == (10, 10)

# %% tags=[]
pc_log2_corr = log2_pc_test_data.loc[_genes_intersect[:10]].T.corr(method="pearson")
display(pc_log2_corr.shape)
display(pc_log2_corr)

assert raw_corr.shape == (10, 10)

# %% tags=[]
_g0, _g1 = "ENSG00000000419.12", "ENSG00000000938.12"
assert raw_corr.loc[_g0, _g1] != pc_log2_corr.loc[_g0, _g1]

# %% [markdown] tags=[]
# The Pearson correlation between raw and pc_log2 for the same pair of genes is not the same, since relationships are not exactly linear after the transformation.

# %% [markdown] tags=[]
# ## Spearman

# %% tags=[]
raw_corr = test_data.loc[_genes_intersect[:10]].T.corr(method="spearman")
display(raw_corr.shape)
display(raw_corr)

assert raw_corr.shape == (10, 10)

# %% tags=[]
pc_log2_corr = log2_pc_test_data.loc[_genes_intersect[:10]].T.corr(method="spearman")
display(pc_log2_corr.shape)
display(pc_log2_corr)

assert raw_corr.shape == (10, 10)

# %% tags=[]
_g0, _g1 = "ENSG00000000419.12", "ENSG00000000938.12"
assert raw_corr.loc[_g0, _g1] == pc_log2_corr.loc[_g0, _g1]

# %% [markdown] tags=[]
# The Spearman correlation between raw and pc_log2 for the same pair of genes is the same, since relationships are still monotonic.

# %% [markdown] tags=[]
# # Select top genes for each tissue data file

# %% [markdown] tags=[]
# Based on the previous findings, I select genes with both strategies `var_raw` and `var_pc_log2`.
#
# Then I save, for both, the raw data (note that I only use the strategies to select genes, not to log-transform the data).

# %% tags=[]
input_files = sorted(list(INPUT_DIR.glob("*.pkl")))
assert len(input_files) == 54, len(input_files)

display(input_files[:5])

# %% [markdown] tags=[]
# ## Run

# %% tags=[]
pbar = tqdm(input_files, ncols=100)

for tissue_data_file in pbar:
    pbar.set_description(tissue_data_file.stem)

    tissue_data = pd.read_pickle(tissue_data_file)

    # var_raw
    top_genes_var = (
        tissue_data.var(axis=1)
        .sort_values(ascending=False)
        .head(N_TOP_GENES_MAX_VARIANCE)
    )
    selected_tissue_data = tissue_data.loc[top_genes_var.index]

    output_filename = f"{tissue_data_file.stem}-var_raw.pkl"
    selected_tissue_data.to_pickle(path=OUTPUT_DIR / output_filename)

    # var_pc_log2
    log2_tissue_data = np.log2(tissue_data + 1)

    top_genes_var = (
        log2_tissue_data.var(axis=1)
        .sort_values(ascending=False)
        .head(N_TOP_GENES_MAX_VARIANCE)
    )
    # save the same raw data, but with genes selected by var_pc_log2
    selected_tissue_data = tissue_data.loc[top_genes_var.index]

    output_filename = f"{tissue_data_file.stem}-var_pc_log2.pkl"
    selected_tissue_data.to_pickle(path=OUTPUT_DIR / output_filename)

# %% [markdown] tags=[]
# ## Testing

# %% tags=[]
_tmp_raw = pd.read_pickle(
    OUTPUT_DIR / "gtex_v8_data_brain_nucleus_accumbens_basal_ganglia-var_raw.pkl"
).sort_index()

_tmp_log2 = pd.read_pickle(
    OUTPUT_DIR / "gtex_v8_data_brain_nucleus_accumbens_basal_ganglia-var_pc_log2.pkl"
).sort_index()

# %% tags=[]
display(_tmp_raw.shape)
assert _tmp_raw.shape == _tmp_log2.shape

# %% tags=[]
assert _tmp_raw.columns.tolist() == _tmp_log2.columns.tolist()

# %% tags=[]
_genes_intersect = sorted(list(set(_tmp_raw.index) & set(_tmp_log2.index)))
assert len(_genes_intersect) == 2926

# %% tags=[]
# make sure the same data is stored
assert _tmp_raw.loc[_genes_intersect].equals(_tmp_log2.loc[_genes_intersect])

# %% [markdown] tags=[]
# ### raw

# %% tags=[]
_tmp_raw.head()

# %% tags=[]
_tmp_desc = _tmp_raw.T.describe()
display(_tmp_desc)

assert _tmp_desc.loc["min"].min() == 0.0
assert _tmp_desc.loc["min"].max() == 37640.0

assert _tmp_desc.loc["max"].min() == 30.66
assert _tmp_desc.loc["max"].max() == 201000.0

# %% [markdown] tags=[]
# ### pc_log2

# %% tags=[]
_tmp_log2.head()

# %% tags=[]
# the stats are different than _tmp_raw
_tmp_desc = _tmp_log2.T.describe()
display(_tmp_desc)

assert _tmp_desc.loc["min"].min() == 0.0
assert _tmp_desc.loc["min"].max() == 201.4

assert _tmp_desc.loc["max"].min() == 5.824
assert _tmp_desc.loc["max"].max() == 15320.0

# %% tags=[]
