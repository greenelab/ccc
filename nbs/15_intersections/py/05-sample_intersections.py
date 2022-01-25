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

# %% tags=[]
DATASET_CONFIG = conf.GTEX
GTEX_TISSUE = "whole_blood"
GENE_SEL_STRATEGY = "var_pc_log2"

# %% tags=[]
# maximum amount of gene pairs to sample
MAX_SAMPLE_SIZE = 1000

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
INPUT_GENE_PAIRS_INTERSECTIONS_FILE = (
    DATASET_CONFIG["GENE_PAIR_INTERSECTIONS"]
    / f"gene_pair_intersections-gtex_v8-{GTEX_TISSUE}-{GENE_SEL_STRATEGY}.pkl"
)
display(INPUT_GENE_PAIRS_INTERSECTIONS_FILE)

assert INPUT_GENE_PAIRS_INTERSECTIONS_FILE.exists()

# %% tags=[]
OUTPUT_FILE = (
    INPUT_GENE_PAIRS_INTERSECTIONS_FILE.parent
    / f"{INPUT_GENE_PAIRS_INTERSECTIONS_FILE.stem}-sample.pkl"
)

display(OUTPUT_FILE)

# %% [markdown] tags=[]
# # Data

# %% [markdown] tags=[]
# ## Gene pairs intersection

# %% tags=[]
df_plot = pd.read_pickle(INPUT_GENE_PAIRS_INTERSECTIONS_FILE)

# %% tags=[]
df_plot.shape

# %% tags=[]
df_plot.head()

# %% [markdown] tags=[]
# # CHANGE - Look at specific gene pair cases

# %% tags=[]
df_r_data = df_plot

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
def get_gene_pairs(first_coef, query_set):
    """
    FIXME: move this function to the library

    It queries a dataframe with the intersections of different groups (i.e.,
    clustermatch high, pearson low, etc) given a query set. It returns a slice of
    the dataframe according to the query set provided.

    The function needs to access a variable named "df_r_data" that has the
    intersections between coefficients.

    Args:
        first_coef: the main coefficient ("clustermatch", "pearson" or "spearman")
            of interest. The final dataframe will be sorted according to this
            coefficient.
        query_set: a tuple with strings that specifies a query. For example
            ("Clustermatch (high)", "Pearson (low") would select all gene pairs
            for which clustermatch is high and pearson is low.

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


# %% [markdown] tags=[]
# # Get intersections dataframe

# %% tags=[]
intersections = {}

# %% [markdown] tags=[]
# ## Agreements

# %% tags=[]
_queries = [
    [
        "Clustermatch (high)",
        "Pearson (high)",
        "Spearman (high)",
        # "Clustermatch (low)",
        # "Pearson (low)",
        # "Spearman (low)",
    ],
    [
        "Clustermatch (high)",
        "Pearson (high)",
        # "Spearman (high)",
        # "Clustermatch (low)",
        # "Pearson (low)",
        # "Spearman (low)",
    ],
    [
        "Clustermatch (high)",
        # "Pearson (high)",
        "Spearman (high)",
        # "Clustermatch (low)",
        # "Pearson (low)",
        # "Spearman (low)",
    ],
    [
        # "Clustermatch (high)",
        "Pearson (high)",
        "Spearman (high)",
        # "Clustermatch (low)",
        # "Pearson (low)",
        # "Spearman (low)",
    ],
    [
        # "Clustermatch (high)",
        # "Pearson (high)",
        # "Spearman (high)",
        "Clustermatch (low)",
        "Pearson (low)",
        # "Spearman (low)",
    ],
    [
        # "Clustermatch (high)",
        # "Pearson (high)",
        # "Spearman (high)",
        "Clustermatch (low)",
        # "Pearson (low)",
        "Spearman (low)",
    ],
    [
        # "Clustermatch (high)",
        # "Pearson (high)",
        # "Spearman (high)",
        # "Clustermatch (low)",
        "Pearson (low)",
        "Spearman (low)",
    ],
    [
        # "Clustermatch (high)",
        # "Pearson (high)",
        # "Spearman (high)",
        "Clustermatch (low)",
        "Pearson (low)",
        "Spearman (low)",
    ],
]

for idx, _query in enumerate(_queries):
    _query_str = str(", ".join(_query))
    assert _query_str not in intersections

    _tmp_df = get_gene_pairs(
        "clustermatch",
        set(_query),
    )

    intersections[_query_str] = _tmp_df.sample(
        n=min(_tmp_df.shape[0], MAX_SAMPLE_SIZE), random_state=idx
    ).index.to_frame(index=False, name=["gene0", "gene1"])

# %% tags=[]
assert len(intersections) == 8

# %% tags=[]
_sizes = set()
for _query_str in intersections.keys():
    _sizes.add(intersections[_query_str].shape[0])

# %% tags=[]
assert _sizes == {MAX_SAMPLE_SIZE}

# %% tags=[]
intersections[_query_str].head()

# %% [markdown] tags=[]
# ## Disagreements

# %% tags=[]
_queries = [
    [
        "Clustermatch (high)",
        # "Pearson (high)",
        "Spearman (high)",
        # "Clustermatch (low)",
        "Pearson (low)",
        # "Spearman (low)",
    ],
    [
        "Clustermatch (high)",
        # "Pearson (high)",
        # "Spearman (high)",
        # "Clustermatch (low)",
        "Pearson (low)",
        # "Spearman (low)",
    ],
    [
        "Clustermatch (high)",
        # "Pearson (high)",
        # "Spearman (high)",
        # "Clustermatch (low)",
        # "Pearson (low)",
        "Spearman (low)",
    ],
    [
        "Clustermatch (high)",
        # "Pearson (high)",
        # "Spearman (high)",
        # "Clustermatch (low)",
        "Pearson (low)",
        "Spearman (low)",
    ],
    [
        # "Clustermatch (high)",
        "Pearson (high)",
        # "Spearman (high)",
        "Clustermatch (low)",
        # "Pearson (low)",
        # "Spearman (low)",
    ],
    [
        # "Clustermatch (high)",
        "Pearson (high)",
        # "Spearman (high)",
        # "Clustermatch (low)",
        # "Pearson (low)",
        "Spearman (low)",
    ],
    [
        # "Clustermatch (high)",
        "Pearson (high)",
        # "Spearman (high)",
        "Clustermatch (low)",
        # "Pearson (low)",
        "Spearman (low)",
    ],
    [
        # "Clustermatch (high)",
        # "Pearson (high)",
        "Spearman (high)",
        # "Clustermatch (low)",
        "Pearson (low)",
        # "Spearman (low)",
    ],
]

for idx, _query in enumerate(_queries):
    _query_str = str(", ".join(_query))
    assert _query_str not in intersections

    _tmp_df = get_gene_pairs(
        "clustermatch",
        set(_query),
    )

    intersections[_query_str] = _tmp_df.sample(
        n=min(_tmp_df.shape[0], MAX_SAMPLE_SIZE), random_state=idx
    ).index.to_frame(index=False, name=["gene0", "gene1"])

# %% tags=[]
assert len(intersections) == 16

# %% tags=[]
_sizes = set()
for _query_str in intersections.keys():
    _sizes.add(intersections[_query_str].shape[0])

# %% tags=[]
_sizes

# %% tags=[]
assert _sizes == {MAX_SAMPLE_SIZE, 28, 8, 87, 531, 527}

# %% tags=[]
intersections[_query_str].head()

# %% [markdown] tags=[]
# # Save

# %% tags=[]
import pickle

# %% tags=[]
with open(OUTPUT_FILE, "wb") as handle:
    pickle.dump(intersections, handle)

# %% tags=[]
