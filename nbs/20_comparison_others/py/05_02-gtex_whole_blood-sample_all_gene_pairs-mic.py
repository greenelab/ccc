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
from pathlib import Path

import pandas as pd
import numpy as np

from clustermatch import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
DATASET_CONFIG = conf.GTEX
GTEX_TISSUE = "whole_blood"
GENE_SEL_STRATEGY = "var_pc_log2"

METHOD_NAME = "mic"

# %% tags=["parameters"]
# this cell has the "parameters" tag

# size of gene pair groups to process in parallel
CHUNK_SIZE = 100

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
GENE_PAIRS_FILE_SUFFIX = (
    DATASET_CONFIG["GENE_SELECTION_DIR"]
    / f"gtex_v8_data_{GTEX_TISSUE}-{GENE_SEL_STRATEGY}.pkl"
)
display(GENE_PAIRS_FILE_SUFFIX)

assert GENE_PAIRS_FILE_SUFFIX.exists()

# %% tags=[]
INPUT_DIR = GENE_PAIRS_FILE_SUFFIX.parent / "samples"
display(INPUT_DIR)

# %% tags=[]
INPUT_GENE_PAIRS_FILE = INPUT_DIR / (
    f"{GENE_PAIRS_FILE_SUFFIX.stem}-gene_pairs-sample_" + "{sample_id}" + ".pkl"
)
display(INPUT_GENE_PAIRS_FILE)

INPUT_GENE_PAIRS_FILE_TEMPLATE = str(INPUT_GENE_PAIRS_FILE)
display(INPUT_GENE_PAIRS_FILE_TEMPLATE)

# %% tags=[]
OUTPUT_DIR = DATASET_CONFIG["RESULTS_DIR"] / "comparison_others" / METHOD_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %% tags=[]
OUTPUT_FILE_TEMPLATE = str(
    OUTPUT_DIR / (INPUT_GENE_PAIRS_FILE.name[:-4] + f"-{METHOD_NAME}.pkl")
)

display(OUTPUT_FILE_TEMPLATE)

# %% [markdown] tags=[]
# # Data

# %% [markdown] tags=[]
# ## Gene expression

# %% tags=[]
gene_expr_dict = pd.read_pickle(INPUT_GENE_EXPR_FILE).T.to_dict(orient="series")

# %% tags=[]
len(gene_expr_dict)

# %% tags=[]
gene_expr_dict[list(gene_expr_dict.keys())[0]]

# %% [markdown] tags=[]
# ## Gene pairs intersection

# %% tags=[]
# intersections = pd.read_pickle(INPUT_GENE_PAIRS_INTERSECTIONS_FILE)

# %% tags=[]
# len(intersections)

# %% tags=[]
# intersections["Clustermatch (high), Pearson (high), Spearman (high)"]

# %% [markdown] tags=[]
# # Compute Maximal Information Coefficient (MIC)

# %% [markdown] tags=[]
# ## Functions

# %% tags=[]
import warnings
from sklearn.metrics import pairwise_distances
from minepy.mine import MINE


# %% tags=[]
def _mic(x, y):
    """
    FIXME: move to library
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        mine = MINE(alpha=0.6, c=15, est="mic_approx")
        mine.compute_score(x, y)
        return mine.mic()


# %% tags=[]
_mic(np.random.rand(10), np.random.rand(10))

# %% [markdown] tags=[]
# ## Get all sample files

# %% tags=[]
all_sample_files = []

sample_id = 0
sample_file = Path(INPUT_GENE_PAIRS_FILE_TEMPLATE.format(sample_id=sample_id))

while sample_file.exists():
    all_sample_files.append((sample_id, sample_file))

    sample_id += 1
    sample_file = Path(INPUT_GENE_PAIRS_FILE_TEMPLATE.format(sample_id=sample_id))

# %% tags=[]
# all_sample_files = sorted(
#     list(
#         INPUT_GENE_PAIRS_FILE.parent.glob(INPUT_GENE_PAIRS_FILE.name.format(sample_id="*"))
#     )
# )

# %% tags=[]
len(all_sample_files)

# %% tags=[]
all_sample_files[:3]

# %% [markdown] tags=[]
# ## Run

# %% tags=[]
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

from tqdm import tqdm

from clustermatch.utils import chunker


# %% tags=[]
def _compute_mic(gene_sets: list):
    res = [
        _mic(gene_expr_dict[gs[0]].to_numpy(), gene_expr_dict[gs[1]].to_numpy())
        for gs in gene_sets
    ]

    return pd.Series(res, index=pd.MultiIndex.from_tuples(gene_sets))


# %% tags=[]
# testing
# gene_set_key = "Clustermatch (high), Pearson (high), Spearman (high)"
gene_set = pd.read_pickle(all_sample_files[0][1]).sample(n=10)
display(gene_set)

_res = _compute_mic(list(gene_set.itertuples(index=False)))
display(_res.shape)
display(_res.head())

# make sure order is preserved
assert _res.index.to_list() == list(gene_set.itertuples(index=False, name=None))

# %% tags=[]
all_chunks = []

for (sample_id, sample_file) in all_sample_files:
    gene_pairs_df = pd.read_pickle(sample_file)
    gene_pairs_subset = list(gene_pairs_df.itertuples(index=False, name=None))

    for chunk in list(chunker(gene_pairs_subset, CHUNK_SIZE)):
        all_chunks.append((sample_id, chunk))

# %% tags=[]
len(all_chunks)

# %% tags=[]
all_chunks[:2]

# %% tags=[]
all_results = defaultdict(list)

with ProcessPoolExecutor(max_workers=conf.GENERAL["N_JOBS"]) as executor:
    tasks = {
        executor.submit(_compute_mic, chunk): sample_id
        for sample_id, chunk in all_chunks
    }

    pbar = tqdm(as_completed(tasks), total=len(all_chunks), ncols=100)

    for future in pbar:
        sample_id = tasks[future]
        sample_file_mic = future.result()

        all_results[sample_id].append(sample_file_mic)

# %% [markdown] tags=[]
# # Save for each sample file

# %% tags=[]
for sample_id in all_results.keys():
    sample_file_all_results_df = pd.concat(all_results[sample_id]).sort_index()
    assert not sample_file_all_results_df.isna().any()
    sample_file_all_results_gene_pairs_set = set(sample_file_all_results_df.index)

    # testing: load input gene pairs
    sample_file_df = pd.read_pickle(
        INPUT_GENE_PAIRS_FILE_TEMPLATE.format(sample_id=sample_id)
    )
    assert sample_file_df.drop_duplicates().shape[0] == sample_file_df.shape[0]

    # testing: number of gene pairs are the same in input data and in results
    assert sample_file_df.shape[0] == sample_file_all_results_df.shape[0]

    # testing: make sure gene ids are the same in results as in input gene pairs
    sample_file_gene_pairs_list = list(
        sample_file_df.itertuples(index=False, name=None)
    )
    sample_file_gene_pairs_set = set(sample_file_gene_pairs_list)
    assert len(sample_file_gene_pairs_set) == len(
        sample_file_gene_pairs_set.intersection(sample_file_all_results_gene_pairs_set)
    )

    # save results with same order (in gene pairs) as input sample data
    sample_file_all_results_df = sample_file_all_results_df.loc[
        sample_file_gene_pairs_list
    ]
    sample_file_all_results_df.to_pickle(
        OUTPUT_FILE_TEMPLATE.format(sample_id=sample_id)
    )

# %% tags=[]
# show how one result set looks like
display(sample_file_all_results_df.shape)
display(sample_file_all_results_df.head())

# %% tags=[]
