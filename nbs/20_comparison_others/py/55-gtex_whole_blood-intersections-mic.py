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
# It analyzes how the CCC and MIC coefficients intersect on different gene pairs. This notebook is very similar to the [other notebook](https://github.com/greenelab/ccc/blob/coef_improvements/nbs/99_manuscript/coefs_comp/08_05-gtex_whole_blood-intersections_plots.ipynb) that compares CCC, Spearman and Pearson.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from upsetplot import plot, from_indicators

from ccc import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
DATASET_CONFIG = conf.GTEX
GTEX_TISSUE = "whole_blood"
GENE_SEL_STRATEGY = "var_pc_log2"

# %% tags=[]
# this specificies the threshold to compare coefficients (see below).
# it basically takes the top Q_DIFF coefficient values for gene pairs
# and compare with the bottom Q_DIFF of the other coefficients
Q_DIFF = 0.30

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
COMPARISONS_DIR = DATASET_CONFIG["RESULTS_DIR"] / "comparison_others"
display(COMPARISONS_DIR)

# %% tags=[]
INPUT_CORR_FILE = (
    COMPARISONS_DIR / f"gtex_v8_data_{GTEX_TISSUE}-{GENE_SEL_STRATEGY}-all.pkl"
)
display(INPUT_CORR_FILE)

assert INPUT_CORR_FILE.exists()

# %% [markdown] tags=[]
# # Data

# %% [markdown] tags=[]
# ## Correlation

# %% tags=[]
df = pd.read_pickle(INPUT_CORR_FILE)

# %% tags=[]
df.shape

# %% tags=[]
df.head()

# %% tags=[]
df.describe().applymap(str)

# %% tags=[]
# show quantiles
df.apply(lambda x: x.quantile(np.linspace(0.20, 1.0, 20)))


# %% [markdown] tags=[]
# # Prepare data for plotting

# %% tags=[]
def get_lower_upper_quantile(method_name, q):
    return df[method_name].quantile([q, 1 - q])


# %% tags=[]
# test
_tmp = get_lower_upper_quantile("ccc", 0.20)
display(_tmp)

_tmp0, _tmp1 = _tmp
display((_tmp0, _tmp1))

assert _tmp0 == _tmp.iloc[0]
assert _tmp1 == _tmp.iloc[1]

# %% tags=[]
clustermatch_lq, clustermatch_hq = get_lower_upper_quantile("ccc", Q_DIFF)
display((clustermatch_lq, clustermatch_hq))

pearson_lq, pearson_hq = get_lower_upper_quantile("pearson", Q_DIFF)
display((pearson_lq, pearson_hq))

spearman_lq, spearman_hq = get_lower_upper_quantile("spearman", Q_DIFF)
display((spearman_lq, spearman_hq))

mic_lq, mic_hq = get_lower_upper_quantile("mic", Q_DIFF)
display((mic_lq, mic_hq))

# %% tags=[]
pearson_higher = df["pearson"] >= pearson_hq
display(pearson_higher.sum())

pearson_lower = df["pearson"] <= pearson_lq
display(pearson_lower.sum())

# %% tags=[]
spearman_higher = df["spearman"] >= spearman_hq
display(spearman_higher.sum())

spearman_lower = df["spearman"] <= spearman_lq
display(spearman_lower.sum())

# %% tags=[]
clustermatch_higher = df["ccc"] >= clustermatch_hq
display(clustermatch_higher.sum())

clustermatch_lower = df["ccc"] <= clustermatch_lq
display(clustermatch_lower.sum())

# %% tags=[]
mic_higher = df["mic"] >= mic_hq
display(mic_higher.sum())

mic_lower = df["mic"] <= mic_lq
display(mic_lower.sum())

# %% [markdown] tags=[]
# **Question:** Why the number of top/bottom gene pairs in CCC does not match the rest? Maybe it's because there are repeated values. Let's see:

# %% tags=[]
df.shape

# %% tags=[]
df["pearson"].unique().shape

# %% tags=[]
df["spearman"].unique().shape

# %% tags=[]
df["ccc"].unique().shape

# %% tags=[]
df["mic"].unique().shape

# %% [markdown] tags=[]
# Yes, many CCC values are the same!

# %% [markdown] tags=[]
# # UpSet plot

# %% tags=[]
df_plot = pd.DataFrame(
    {
        "pearson_higher": pearson_higher,
        "pearson_lower": pearson_lower,
        "spearman_higher": spearman_higher,
        "spearman_lower": spearman_lower,
        "clustermatch_higher": clustermatch_higher,
        "clustermatch_lower": clustermatch_lower,
        "mic_higher": mic_higher,
        "mic_lower": mic_lower,
    }
)

# %% tags=[]
df_plot = pd.concat([df_plot, df], axis=1)

# %% tags=[]
df_plot

# %% tags=[]
assert not df_plot.isna().any().any()

# %% tags=[]
df_plot = df_plot.rename(
    columns={
        "pearson_higher": "Pearson (high)",
        "pearson_lower": "Pearson (low)",
        "spearman_higher": "Spearman (high)",
        "spearman_lower": "Spearman (low)",
        "clustermatch_higher": "CCC (high)",
        "clustermatch_lower": "CCC (low)",
        "mic_higher": "MIC (high)",
        "mic_lower": "MIC (low)",
    }
)

# %% tags=[]
categories = sorted(
    [x for x in df_plot.columns if " (" in x],
    reverse=True,
    key=lambda x: x.split(" (")[1] + " (" + x.split(" (")[0],
)

# %% tags=[]
categories

# %% [markdown] tags=[]
# ## All subsets (original full plot)

# %% tags=[]
df_r_data = df_plot

# %% tags=[]
df_r_data.shape

# %% tags=[]
gene_pairs_by_cats = from_indicators(categories, data=df_r_data)

# %% tags=[]
gene_pairs_by_cats

# %% tags=[]
fig = plt.figure(figsize=(18, 5))

g = plot(
    gene_pairs_by_cats,
    show_counts=True,
    sort_categories_by=None,
    element_size=None,
    fig=fig,
)

# %% [markdown] tags=[]
# There are no intersections (columns) where CCC is high/low and MIC is low/high. This indicates that both behave very similarly.
#
# To make this more clear, below I only focus on MIC and CCC.

# %% [markdown] tags=[]
# ## Only CCC and MIC

# %% tags=[]
categories = sorted(
    [x for x in df_plot.columns if " (" in x and ("CCC" in x or "MIC" in x)],
    reverse=True,
    key=lambda x: x.split(" (")[1] + " (" + x.split(" (")[0],
)

# %% tags=[]
categories

# %% tags=[]
df_r_data = df_plot

# %% tags=[]
df_r_data.shape

# %% tags=[]
gene_pairs_by_cats = from_indicators(categories, data=df_r_data)

# %% tags=[]
gene_pairs_by_cats

# %% tags=[]
fig = plt.figure(figsize=(18, 5))

g = plot(
    gene_pairs_by_cats,
    show_counts=True,
    sort_categories_by=None,
    element_size=None,
    fig=fig,
)

# %% tags=[]
