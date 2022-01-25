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
# TODO

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import pandas as pd
import numpy as np

from clustermatch import conf

# %% [markdown] tags=[]
# # Settings

# %%
DATASET_CONFIG = conf.GTEX
GTEX_TISSUE = "whole_blood"
GENE_SEL_STRATEGY = "var_pc_log2"

# %% tags=["parameters"]
# this cell has the "parameters" tag

# size of gene pair groups to process in parallel
CHUNK_SIZE = 50

# %% [markdown] tags=[]
# # Paths

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
    / f"gene_pair_intersections-gtex_v8-{GTEX_TISSUE}-{GENE_SEL_STRATEGY}-sample.pkl"
)
display(INPUT_GENE_PAIRS_INTERSECTIONS_FILE)

assert INPUT_GENE_PAIRS_INTERSECTIONS_FILE.exists()

# %%
OUTPUT_FILE = (
    INPUT_GENE_PAIRS_INTERSECTIONS_FILE.parent
    / f"{INPUT_GENE_PAIRS_INTERSECTIONS_FILE.stem}-mic.pkl"
)

display(OUTPUT_FILE)

# %% [markdown] tags=[]
# # Data

# %% [markdown] tags=[]
# ## Gene expression

# %%
gene_expr_dict = pd.read_pickle(INPUT_GENE_EXPR_FILE).T.to_dict(orient="series")

# %%
len(gene_expr_dict)

# %%
gene_expr_dict[list(gene_expr_dict.keys())[0]]

# %% [markdown] tags=[]
# ## Gene pairs intersection

# %%
intersections = pd.read_pickle(INPUT_GENE_PAIRS_INTERSECTIONS_FILE)

# %%
len(intersections)

# %%
intersections["Clustermatch (high), Pearson (high), Spearman (high)"]

# %% [markdown]
# # Compute Maximal Information Coefficient (MIC)

# %% [markdown]
# ## Functions

# %%
import warnings
from sklearn.metrics import pairwise_distances
from minepy.mine import MINE


# %%
def _mic(x, y):
    """
    FIXME: move to library
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        mine = MINE(alpha=0.6, c=15, est="mic_approx")
        mine.compute_score(x, y)
        return mine.mic()


# %%
_mic(np.random.rand(10), np.random.rand(10))

# %% [markdown]
# ## Run

# %%
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

from tqdm import tqdm

from clustermatch.utils import chunker


# %%
def _compute_mic(gene_sets):
    res = {
        (gs[0], gs[1]): _mic(
            gene_expr_dict[gs[0]].to_numpy(), gene_expr_dict[gs[1]].to_numpy()
        )
        for gs in gene_sets
    }

    return pd.Series(res, index=gene_sets)


# %%
# testing
gene_set_key = "Clustermatch (high), Pearson (high), Spearman (high)"
gene_set = intersections[gene_set_key].sample(n=10)

_res = _compute_mic(list(gene_set.itertuples(index=False)))
assert _res.index.to_list() == list(gene_set.itertuples(index=False, name=None))

# %%
all_chunks = []

for (
    gene_set_key
) in intersections.keys():  # ["Clustermatch (high), Pearson (low), Spearman (low)"]
    gene_set = list(intersections[gene_set_key].itertuples(index=False, name=None))

    for chunk in list(chunker(list(gene_set), CHUNK_SIZE)):
        all_chunks.append((gene_set_key, chunk))

# all_chunks = [
#     (gene_set_key, chunk)
#     for chunk in list(chunker(list(intersections[gene_set_key].itertuples(index=False, name=None)), 2))
#     for gene_set_key in ["Clustermatch (high), Pearson (low), Spearman (low)"] # intersections.keys()
# ]

# %%
len(all_chunks)

# %%
all_chunks[:1]

# %%
all_results = defaultdict(list)

with ProcessPoolExecutor(max_workers=conf.GENERAL["N_JOBS"]) as executor:
    tasks = {
        executor.submit(_compute_mic, chunk): gene_set_key
        for gene_set_key, chunk in all_chunks
    }

    pbar = tqdm(as_completed(tasks), total=len(all_chunks), ncols=100)

    for future in pbar:
        gene_set_key = tasks[future]
        gene_set_mic = future.result()

        all_results[gene_set_key].append(gene_set_mic)

_tmp = {}

for k in all_results.keys():
    _tmp[k] = pd.concat(all_results[k])

all_results = _tmp

# %%
assert len(all_results) == len(intersections.keys())

# %% [markdown] tags=[]
# # Save

# %%
import pickle

# %%
with open(OUTPUT_FILE, "wb") as handle:
    pickle.dump(all_results, handle)

# %%
