# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# It analyzes different strategies to take the genes from GTEx data with the highest variability, being this variability measured with different strategies: variance (`var`), coefficient of variation (`cv`) and mean absolute variation (`mad`) applied on two different versions of the data: 1) the raw TPM-normalized gene expression data (here refered to as `raw`), and 2) the log2-transformed version of the raw data (here refered to as `log2`).

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

# %% [markdown] tags=[]
# ## Get test data in log2

# %% tags=[]
log2_test_data = np.log2(test_data)

# %% tags=[]
log2_test_data.head()


# %% tags=[]
def replace_by_minimum(sample_data):
    """Replaces the -np.inf values in a pandas series by [the minimum non-inf value in it] * 1.3."""

    sample_min_values = sample_data.replace(-np.inf, np.nan).dropna().sort_values()
    sample_min = sample_min_values.iloc[0]

    return sample_data.replace(-np.inf, sample_min * 1.3)


# %% tags=[]
assert (
    log2_test_data.iloc[:, [0]]
    .apply(replace_by_minimum)
    .squeeze()
    .loc["ENSG00000278267.1"]
    .round(5)
    == -14.76284
)

# %% tags=[]
log2_test_data = log2_test_data.apply(replace_by_minimum)

# %% tags=[]
log2_test_data.shape

# %% tags=[]
log2_test_data.head()

# %% tags=[]
log2_test_data.iloc[:10, :].T.describe()

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
assert len(genes_selection_methods) == 6

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
# Some methods select very different sets of genes, particularly between `cv`, where there is no agreement between the same approach on `log2` and `raw` data.
#
# Since they are similar, the largest overlap is between `var_*` anad `mad_*` approaches.

# %% tags=[]
# get list of methods that agree more with the rest
(_tmp.sum() - 5000).sort_values(ascending=False)

# %% [markdown] tags=[]
# # How different are genes selected by `raw` and `log2`?

# %% [markdown] tags=[]
# Here I focus on `raw` and `log2` with the `var` (variance) method.

# %% tags=[]
genes_df = pd.DataFrame(top_genes_var)

# %% tags=[]
genes_df.shape

# %% tags=[]
genes_df.head()

# %% tags=[]
cols = ["var_raw", "var_log2"]

# %% [markdown] tags=[]
# ## Genes select in both raw and log2

# %% tags=[]
# show top genes selected by both var_raw and var_log2
genes_df.loc[
    top_genes_var["var_raw"].index.intersection(top_genes_var["var_log2"].index), cols
].head()

# %% tags=[]
_gene_ids = [
    "ENSG00000163631.16",  # larger in raw
    "ENSG00000110245.11",  # larger in log2
]

# %% tags=[]
# plot density on raw
for gene_id in _gene_ids:
    sns.kdeplot(data=test_data.T, x=gene_id, label=gene_id)

plt.legend()

# %% tags=[]
# same genes, but plot density on log2
for gene_id in _gene_ids:
    sns.kdeplot(data=log2_test_data.T, x=gene_id, label=gene_id)

plt.legend()

# %% [markdown] tags=[]
# `var_log2` seems to select genes that tend to be more bimodal (with many cases around no expression and other around highly expressed), whereas `var_raw` selects genes with a unimodal distribution.

# %% [markdown] tags=[]
# ## Genes selected in raw only

# %% tags=[]
# show top genes selected by var_raw
genes_df.loc[top_genes_var["var_raw"].index, cols].head()

# %% tags=[]
_gene_ids = ["ENSG00000244734.3", "ENSG00000188536.12"]

# %% tags=[]
# plot density on raw
for gene_id in _gene_ids:
    sns.kdeplot(data=test_data.T, x=gene_id, label=gene_id)

plt.legend()

# %% tags=[]
# same genes, but plot density on log2
for gene_id in _gene_ids:
    sns.kdeplot(data=log2_test_data.T, x=gene_id, label=gene_id)

plt.legend()

# %% [markdown] tags=[]
# ## Genes selected in log2 only

# %% tags=[]
# show top genes selected by var_log2
genes_df.loc[top_genes_var["var_log2"].index, cols].head()

# %% tags=[]
_gene_ids = ["ENSG00000213058.3", "ENSG00000200879.1", "ENSG00000211918.1"]

# %% tags=[]
# plot density on raw
for gene_id in _gene_ids:
    sns.kdeplot(data=test_data.T, x=gene_id, label=gene_id)

plt.legend()

# %% tags=[]
# same genes, but plot density on log2
for gene_id in _gene_ids:
    sns.kdeplot(data=log2_test_data.T, x=gene_id, label=gene_id)

plt.legend()

# %% [markdown] tags=[]
# **CONCLUSION:** Both `var_raw` (that is, the strategy that selects the top genes with highest variance on raw TPM-normalized data) and `var_log2` (highest variance on log2-transformed TPM-normalized data) seem to be interesting. `var_raw` seems to select genes that are expressed around a mean, less expressed in some conditions and more expressed in others. `var_log2` tends to select genes that are not expressed (zero expression) in several conditions, and relatively highly expressed in others, which might capture important genes such as transcriptor factors (see https://www.biorxiv.org/content/10.1101/2020.02.13.944777v1).

# %% [markdown] tags=[]
# # Select top genes for each tissue data file

# %% [markdown] tags=[]
# Based on the previous findings, I select genes with both strategies `var_raw` and `var_log2`.

# %% tags=[]
input_files = list(INPUT_DIR.glob("*.pkl"))
assert len(input_files) == 54, len(input_files)

display(input_files[:5])

# %% [markdown] tags=[]
# ## Run

# %% tags=[]
pbar = tqdm(input_files, ncols=100)

for tissue_data_file in pbar:
    pbar.set_description(tissue_data_file.stem)

    tissue_data = pd.read_pickle(tissue_data_file)

    # select top genes

    ## var_raw
    top_genes_var = (
        tissue_data.var(axis=1)
        .sort_values(ascending=False)
        .head(N_TOP_GENES_MAX_VARIANCE)
    )
    selected_tissue_data = tissue_data.loc[top_genes_var.index]

    output_filename = f"{tissue_data_file.stem}-var_raw.pkl"
    selected_tissue_data.to_pickle(path=OUTPUT_DIR / output_filename)

    ## var_log2
    log2_tissue_data = np.log2(tissue_data)
    log2_tissue_data = log2_tissue_data.apply(replace_by_minimum)

    top_genes_var = (
        log2_tissue_data.var(axis=1)
        .sort_values(ascending=False)
        .head(N_TOP_GENES_MAX_VARIANCE)
    )
    selected_tissue_data = tissue_data.loc[top_genes_var.index]

    output_filename = f"{tissue_data_file.stem}-var_log2.pkl"
    selected_tissue_data.to_pickle(path=OUTPUT_DIR / output_filename)

# %% [markdown] tags=[]
# ## Testing

# %% tags=[]
_tmp_raw = pd.read_pickle(
    OUTPUT_DIR / "gtex_v8_data_brain_nucleus_accumbens_basal_ganglia-var_raw.pkl"
)
_tmp_log2 = pd.read_pickle(
    OUTPUT_DIR / "gtex_v8_data_brain_nucleus_accumbens_basal_ganglia-var_log2.pkl"
)

# %% tags=[]
display(_tmp_raw.shape)
assert _tmp_raw.shape == _tmp_log2.shape

# %% tags=[]
_tmp_raw.head()

# %% tags=[]
_tmp_desc = _tmp_raw.T.iloc[:, :5].describe()
display(_tmp_desc)

assert _tmp_desc.loc["max"].min() > 80000
assert _tmp_desc.loc["max"].min() < 205000

# %% tags=[]
_tmp_log2.head()

# %% tags=[]
_tmp_desc = _tmp_log2.T.iloc[:, :5].describe()
display(_tmp_desc)

assert _tmp_desc.loc["max"].min() > 8
assert _tmp_desc.loc["max"].min() < 300

# %% tags=[]
assert _tmp_raw.columns.tolist() == _tmp_log2.columns.tolist()

# %% tags=[]
assert len(set(_tmp_raw.index).intersection(set(_tmp_log2.index))) == 23

# %% tags=[]
