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

# %%
import numpy as np
from scipy import stats
from scipy.spatial.distance import squareform
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from clustermatch import conf
from clustermatch.coef import cm

# %%
INPUT_DIR = conf.DATA_DIR / "gan_gene_expr"
display(INPUT_DIR)

# %% [markdown]
# # Load data

# %% [markdown]
# ## Top common genes

# %%
common_top_genes = pd.read_pickle(
    INPUT_DIR / "cm_gen_test_top_10k_common_genes.pkl"
).to_list()

# %%
display(common_top_genes[:5])
display(len(common_top_genes))

# %% [markdown]
# ## Gene expression

# %%
gen_data = pd.read_csv(INPUT_DIR / "gen.csv.gz", usecols=common_top_genes).T

# %%
gen_data.shape

# %%
gen_data.head()

# %%
train_data = pd.read_csv(INPUT_DIR / "train.csv.gz", usecols=common_top_genes).T

# %%
train_data.shape

# %%
train_data.head()

# %%
test_data = pd.read_csv(INPUT_DIR / "test.csv.gz", usecols=common_top_genes).T

# %%
test_data.shape

# %%
test_data.head()

# %%
assert gen_data.index.equals(train_data.index)

# %%
assert gen_data.index.equals(test_data.index)

# %%
assert set(gen_data.index) == set(common_top_genes)

# %% [markdown]
# ## CM correlations

# %%
gen_cm = pd.read_pickle(INPUT_DIR / "cm_gen_top_10k_genes.pkl")

# %%
gen_cm.shape

# %%
gen_cm.head()

# %%
test_cm = pd.read_pickle(INPUT_DIR / "cm_test_top_10k_genes.pkl")

# %%
test_cm.shape

# %%
test_cm.head()

# %%
assert gen_cm.shape[0] == test_cm.shape[0]

# %% [markdown]
# ### Re-index

# %%
# level0 = []
# level1 = []

# for idx0 in range(len(common_top_genes)-1):
#     for idx1 in range(idx0+1, len(common_top_genes)):
#         level0.append(common_top_genes[idx0])
#         level1.append(common_top_genes[idx1])

# %%
# genes_dtype = pd.CategoricalDtype(common_top_genes)

# %%
# level0 = pd.Series(level0, dtype=genes_dtype)

# %%
# level0

# %%
# level1 = pd.Series(level1, dtype=genes_dtype)

# %%
# level1

# %%
new_index = [
    (common_top_genes[idx0], common_top_genes[idx1])
    for idx0 in range(len(common_top_genes) - 1)
    for idx1 in range(idx0 + 1, len(common_top_genes))
]

# %%
display(new_index[:5])
display(len(new_index))

assert len(new_index) == test_cm.shape[0]

# %%
new_index = pd.MultiIndex.from_tuples(new_index)

# new_index = pd.MultiIndex.from_arrays([level0, level1], names=("gene1", "gene2"))

# %%
gen_cm.index = new_index.copy()

# %%
gen_cm

# %%
test_cm.index = new_index.copy()

# %%
test_cm

# %%
assert gen_cm.index.equals(test_cm.index)

# %%
coefs = pd.DataFrame(
    {
        "gen": gen_cm,
        "test": test_cm,
    }
)

# %%
coefs.shape

# %%
coefs

# %% [markdown]
# # Compare

# %%
QUANTILES = np.linspace(0, 1, 10000)
display(QUANTILES[:10])
display(QUANTILES[-10:])

# %%
quantiles_df = pd.DataFrame(
    {
        "gen": coefs["gen"].quantile(QUANTILES).to_numpy(),
        "test": coefs["test"].quantile(QUANTILES).to_numpy(),
    }
)

# %%
quantiles_df.shape

# %%
quantiles_df.head()

# %%
fig, ax = plt.subplots(figsize=(10, 8))

sns.scatterplot(
    data=quantiles_df,
    x="test",
    y="gen",
    ax=ax,
)

# ax.set_xlabel(None)
# ax.set_ylabel("clustermatch (linear)")

min_val = min((quantiles_df.iloc[:, 0].min(), quantiles_df.iloc[:, 1].min()))
max_val = max((quantiles_df.iloc[:, 0].max(), quantiles_df.iloc[:, 1].max()))
ax.plot([min_val, max_val], [min_val, max_val], "k", linewidth=0.5)

# ax.set_title(f"{ENRICH_FUNC} ({PERFORMANCE_MEASURE})")

# %%
# cm(gen_cm.to_numpy(), test_cm.to_numpy())

# %%
stats.pearsonr(gen_cm, test_cm)

# %%
sns.jointplot(
    data=coefs,
    x="gen",
    y="test",
    kind="hex",
    bins="log",
)

# %%
with sns.plotting_context("talk", font_scale=1.1):
    fig, ax = plt.subplots(figsize=(10, 7))

    for method in [x for x in coefs.columns]:
        sns.distplot(x=coefs[method], hist=False, kde=True, label=method, ax=ax)

    plt.legend()

# %%
with sns.plotting_context("talk", font_scale=1.1):
    fig, ax = plt.subplots(figsize=(10, 7))

    for method in [x for x in coefs.columns]:
        sns.distplot(x=coefs[method], hist=True, kde=False, label=method, ax=ax)

    plt.legend()

# %%
coefs.describe()

# %%
_start, _step, _n = 0.10, 0.10, 9
coefs_q = coefs.quantile(np.linspace(_start, _start + (_step * _n), _n, endpoint=False))
display(coefs_q)

# %% [markdown]
# # Interesting gene pairs

# %%
_tmp_coefs = coefs.assign(diff=coefs["test"].sub(coefs["gen"])).sort_values(
    "diff", ascending=False
)

# %%
_tmp_coefs.head(20)

# %%
gene0, gene1 = _tmp_coefs.iloc[2].name
display((gene0, gene1))

# _clustermatch = df.loc[(gene0, gene1), ["pearson", "spearman", "clustermatch"]].tolist()

# %%
# _title = f"Clustermatch: {_clustermatch:.2f}\nPearson/Spearman: {_pearson:.2f}/{_spearman:.2f}"

# %%
p = sns.jointplot(
    data=gen_data.T,
    x=gene0,
    y=gene1,
    kind="hex",
    bins="log",
)

# gene_x_id = p.ax_joint.get_xlabel()
# gene_x_symbol = gene_map[gene_x_id]
# p.ax_joint.set_xlabel(f"{gene_x_id}\n{gene_x_symbol}")

# gene_y_id = p.ax_joint.get_ylabel()
# gene_y_symbol = gene_map[gene_y_id]
# p.ax_joint.set_ylabel(f"{gene_y_id}\n{gene_y_symbol}")

# p.fig.suptitle(_title)

# %%
p = sns.jointplot(
    data=test_data.T,
    x=gene0,
    y=gene1,
    kind="hex",
    bins="log",
)

# gene_x_id = p.ax_joint.get_xlabel()
# gene_x_symbol = gene_map[gene_x_id]
# p.ax_joint.set_xlabel(f"{gene_x_id}\n{gene_x_symbol}")

# gene_y_id = p.ax_joint.get_ylabel()
# gene_y_symbol = gene_map[gene_y_id]
# p.ax_joint.set_ylabel(f"{gene_y_id}\n{gene_y_symbol}")

# p.fig.suptitle(_title)

# %%
_tmp = test_data.T[[gene0, gene1]]

display(_tmp.corr())

display(cm(_tmp.T.to_numpy()))

display(_tmp.shape)
display(_tmp.describe())

# %%
p = sns.jointplot(
    data=train_data.T,
    x=gene0,
    y=gene1,
    kind="hex",
    bins="log",
)

# gene_x_id = p.ax_joint.get_xlabel()
# gene_x_symbol = gene_map[gene_x_id]
# p.ax_joint.set_xlabel(f"{gene_x_id}\n{gene_x_symbol}")

# gene_y_id = p.ax_joint.get_ylabel()
# gene_y_symbol = gene_map[gene_y_id]
# p.ax_joint.set_ylabel(f"{gene_y_id}\n{gene_y_symbol}")

# p.fig.suptitle(_title)

# %%
_tmp = train_data.T[[gene0, gene1]]

display(_tmp.corr())

display(cm(_tmp.T.to_numpy()))

display(_tmp.shape)
display(_tmp.describe())

# %% [markdown]
# # Test is higher

# %%
# gen higher
_tmp_test_higher = coefs[
    (coefs["test"] >= coefs_q.loc[0.90, "test"])
    & (coefs["gen"] <= coefs_q.loc[0.10, "gen"])
].sort_values("test", ascending=False)

display(_tmp_test_higher.shape)
display(_tmp_test_higher)

# %%
gene0, gene1 = _tmp_test_higher.iloc[26].name
display((gene0, gene1))

# _clustermatch = df.loc[(gene0, gene1), ["pearson", "spearman", "clustermatch"]].tolist()

# %%
# _title = f"Clustermatch: {_clustermatch:.2f}\nPearson/Spearman: {_pearson:.2f}/{_spearman:.2f}"

# %%
p = sns.jointplot(
    data=test_data.T,
    x=gene0,
    y=gene1,
    kind="hex",
    bins="log",
)

# gene_x_id = p.ax_joint.get_xlabel()
# gene_x_symbol = gene_map[gene_x_id]
# p.ax_joint.set_xlabel(f"{gene_x_id}\n{gene_x_symbol}")

# gene_y_id = p.ax_joint.get_ylabel()
# gene_y_symbol = gene_map[gene_y_id]
# p.ax_joint.set_ylabel(f"{gene_y_id}\n{gene_y_symbol}")

# p.fig.suptitle(_title)

# %%
_tmp = test_data.T[[gene0, gene1]]

display(_tmp.corr())

display(cm(_tmp.T.to_numpy(), internal_n_clusters=None))

display(_tmp.shape)
display(_tmp.describe())

# %%
p = sns.jointplot(
    data=gen_data.T,
    x=gene0,
    y=gene1,
    kind="hex",
    bins="log",
)

# gene_x_id = p.ax_joint.get_xlabel()
# gene_x_symbol = gene_map[gene_x_id]
# p.ax_joint.set_xlabel(f"{gene_x_id}\n{gene_x_symbol}")

# gene_y_id = p.ax_joint.get_ylabel()
# gene_y_symbol = gene_map[gene_y_id]
# p.ax_joint.set_ylabel(f"{gene_y_id}\n{gene_y_symbol}")

# p.fig.suptitle(_title)

# %%
_tmp = gen_data.T[[gene0, gene1]]

display(_tmp.corr())

display(cm(_tmp.T.to_numpy(), internal_n_clusters=None))

display(_tmp.shape)
display(_tmp.describe())

# %% [markdown]
# # Gen is higher

# %%
# gen higher
_tmp_gen_higher = coefs[
    (coefs["gen"] >= coefs_q.loc[0.90, "gen"])
    & (coefs["test"] <= coefs_q.loc[0.10, "test"])
].sort_values("gen", ascending=False)

display(_tmp_gen_higher.shape)
display(_tmp_gen_higher)

# %%
gene0, gene1 = _tmp_gen_higher.iloc[2].name
display((gene0, gene1))

# _clustermatch = df.loc[(gene0, gene1), ["pearson", "spearman", "clustermatch"]].tolist()

# %%
# _title = f"Clustermatch: {_clustermatch:.2f}\nPearson/Spearman: {_pearson:.2f}/{_spearman:.2f}"

# %%
p = sns.jointplot(
    data=gen_data.T,
    x=gene0,
    y=gene1,
    kind="hex",
    bins="log",
)

# gene_x_id = p.ax_joint.get_xlabel()
# gene_x_symbol = gene_map[gene_x_id]
# p.ax_joint.set_xlabel(f"{gene_x_id}\n{gene_x_symbol}")

# gene_y_id = p.ax_joint.get_ylabel()
# gene_y_symbol = gene_map[gene_y_id]
# p.ax_joint.set_ylabel(f"{gene_y_id}\n{gene_y_symbol}")

# p.fig.suptitle(_title)

# %%
_tmp = gen_data.T[[gene0, gene1]]

display(_tmp.corr())

display(cm(_tmp.T.to_numpy(), internal_n_clusters=None))

display(_tmp.shape)
display(_tmp.describe())

# %%
p = sns.jointplot(
    data=test_data.T,
    x=gene0,
    y=gene1,
    kind="hex",
    bins="log",
)

# gene_x_id = p.ax_joint.get_xlabel()
# gene_x_symbol = gene_map[gene_x_id]
# p.ax_joint.set_xlabel(f"{gene_x_id}\n{gene_x_symbol}")

# gene_y_id = p.ax_joint.get_ylabel()
# gene_y_symbol = gene_map[gene_y_id]
# p.ax_joint.set_ylabel(f"{gene_y_id}\n{gene_y_symbol}")

# p.fig.suptitle(_title)

# %%
_tmp = test_data.T[[gene0, gene1]]

display(_tmp.corr())

display(cm(_tmp.T.to_numpy(), internal_n_clusters=None))

display(_tmp.shape)
display(_tmp.describe())

# %% [markdown]
# ## Other

# %%
gene0, gene1 = "KDM6A", "UTY"
display((gene0, gene1))

# %%
p = sns.jointplot(
    data=gen_data.T,
    x=gene0,
    y=gene1,
    kind="hex",
    bins="log",
)

# gene_x_id = p.ax_joint.get_xlabel()
# gene_x_symbol = gene_map[gene_x_id]
# p.ax_joint.set_xlabel(f"{gene_x_id}\n{gene_x_symbol}")

# gene_y_id = p.ax_joint.get_ylabel()
# gene_y_symbol = gene_map[gene_y_id]
# p.ax_joint.set_ylabel(f"{gene_y_id}\n{gene_y_symbol}")

# p.fig.suptitle(_title)

# %%
