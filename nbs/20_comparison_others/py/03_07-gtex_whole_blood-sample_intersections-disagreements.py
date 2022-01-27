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
# From the intersections groups (visualized in the UpSet plot), it takes all gene pairs from the "Disagreements" groups and saves them.
#
# This notebook does not sample, since the "disagreements" group is small, but I keep the same for convienience.

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
# amount of gene pairs to sample
SAMPLE_SIZE = 30000

# number of samples to take
N_SAMPLES = 10

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
OUTPUT_DIR = INPUT_GENE_PAIRS_INTERSECTIONS_FILE.parent / "samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %% tags=[]
OUTPUT_FILE_TEMPLATE = str(
    OUTPUT_DIR
    / (
        f"{INPUT_GENE_PAIRS_INTERSECTIONS_FILE.stem}-disagreements_sample_"
        + "{sample_id}"
        + ".pkl"
    )
)

display(OUTPUT_FILE_TEMPLATE)


# %% [markdown] tags=[]
# # Data

# %% [markdown] tags=[]
# ## Gene pairs intersection data

# %% tags=[]
def is_intersection_column(column_name):
    return " (high)" in column_name or " (low)" in column_name


# %%
gene_pairs_intersections = pd.read_pickle(INPUT_GENE_PAIRS_INTERSECTIONS_FILE)

# %%
gene_pairs_intersections = gene_pairs_intersections[
    [c for c in gene_pairs_intersections.columns if is_intersection_column(c)]
]

# %%
gene_pairs_intersections.shape

# %%
gene_pairs_intersections.head()


# %% [markdown] tags=[]
# ## Gene pairs intersection - gene pairs

# %% tags=[]
# gene_pairs_df = gene_pairs_intersections.rename_axis(("gene0", "gene1")).index.to_frame(index=False)

# %% tags=[]
# gene_pairs_df.shape

# %% tags=[]
# gene_pairs_df.head()

# %% [markdown] tags=[]
# # Functions

# %% tags=[]
# df_r_data = df_plot

# %% tags=[]
# df_r_data.head()

# %% tags=[]
# df_r_data_boolean_cols = set(
#     [x for x in df_r_data.columns if " (high)" in x or " (low)" in x]
# )

# %% tags=[]
# df_r_data_boolean_cols

# %% tags=[]
def get_gene_pairs(gene_pairs_intersections, query_set):
    """
    FIXME: UPDATE

    It queries the given dataframe with the intersections of different groups (i.e.,
    clustermatch high, pearson low, etc) given a query set. It returns a slice of
    the dataframe according to the query set provided.

    Args:
        gene_pairs_intersections: a dataframe with gene pairs in rows and intersection
            groups as columns (which are boolean). No other columns are allowed.
        query_set: a tuple with strings that specifies a query. For example
            ("Clustermatch (high)", "Pearson (low") would select all gene pairs
            for which clustermatch is high and pearson is low.

    Returns:
        A slice of gene pairs in the input dataframe where the conditions specified in query_set
        apply.
    """
    assert all([x in gene_pairs_intersections.columns for x in query_set])

    query = np.concatenate(
        [
            # columns that have to be true
            np.concatenate(
                [
                    gene_pairs_intersections[c].to_numpy().reshape(-1, 1)
                    for c in query_set
                ],
                axis=1,
            )
            .all(axis=1)
            .reshape(-1, 1),
            # rest of the columns, that have to be false
            np.concatenate(
                [
                    ~gene_pairs_intersections[c].to_numpy().reshape(-1, 1)
                    for c in gene_pairs_intersections.columns
                    if c not in query_set
                ],
                axis=1,
            )
            .all(axis=1)
            .reshape(-1, 1),
        ],
        axis=1,
    ).all(axis=1)

    _tmp_df = gene_pairs_intersections[query]

    # _tmp_df = _tmp_df[
    #     [x for x in _tmp_df.columns if "(high)" not in x and "(low)" not in x]
    # ]

    return _tmp_df.rename_axis(("gene0", "gene1")).index.to_frame(index=False)


# %%
_tmp = get_gene_pairs(
    gene_pairs_intersections,
    {
        "Clustermatch (high)",
        "Pearson (high)",
        "Spearman (high)",
    },
)

display(_tmp.dtypes)
display(_tmp.shape)
display(_tmp.head())

assert _tmp.shape[0] > int(3.12e6)

# %% [markdown] tags=[]
# # Get intersections dataframe

# %% [markdown] tags=[]
# ## Disagreements

# %%
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

# %%
display(len(_queries))
assert len(_queries) == 8

# %% tags=[]
gene_pairs_df = []

for idx, _query in enumerate(_queries):
    group_df = get_gene_pairs(gene_pairs_intersections, set(_query))
    gene_pairs_df.append(group_df)

gene_pairs_df = pd.concat(gene_pairs_df)

# %%
display(gene_pairs_df.shape)
assert gene_pairs_df.drop_duplicates().shape == gene_pairs_df.shape
assert (gene_pairs_df.shape[0] > 3.3e4) and (gene_pairs_df.shape[0] < 3.4e4)

# %%
gene_pairs_df.head()

# %% [markdown] tags=[]
# # Save

# %% [markdown] tags=[]
# Since this "disagreements" set is small, just save the entire group.

# %%
output_filepath = OUTPUT_FILE_TEMPLATE.format(sample_id=0)
display(output_filepath)

# %%
gene_pairs_df.to_pickle(output_filepath)

# %%
