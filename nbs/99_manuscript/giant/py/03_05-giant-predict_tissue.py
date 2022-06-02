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
# Read the top gene pairs prioritized by each correlation coefficient and predict a tissue-specific network using the web services provided by GIANT/HumanBase (https://hb.flatironinstitute.org/). Then it saves the network in files for later processing.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import pandas as pd
from tqdm import tqdm

from ccc import conf
from ccc.giant import get_network

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
DATASET_CONFIG = conf.GTEX

# %% tags=[]
N_TOP_GENE_PAIRS = 100

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
INPUT_DIR = DATASET_CONFIG["GENE_PAIR_INTERSECTIONS"]
display(INPUT_DIR)

assert INPUT_DIR.exists()

# %% tags=[]
OUTPUT_DIR = conf.GIANT["RESULTS_DIR"] / "intersection_genes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %%
TISSUE_SPECIFIC_URLS = {
    "blood": ("blood", "http://hb.flatironinstitute.org/api/integrations/blood/"),
}

# %% [markdown] tags=[]
# # Load gene maps

# %% [markdown]
# These gene mappings include only query genes (gene pairs).

# %%
gene_id_mappings = pd.read_pickle(OUTPUT_DIR / "gene_map-symbol_to_entrezid.pkl")

# %%
gene_id_mappings.shape

# %%
gene_id_mappings.head()

# %%
gene_symbol_to_entrezid = gene_id_mappings.set_index("SYMBOL").squeeze().to_dict()

# %%
gene_entrezid_to_symbol = gene_id_mappings.set_index("ENTREZID").squeeze().to_dict()

# %%
gene_id_mappings.set_index("SYMBOL").loc["ZDHHC12"]


# %% [markdown]
# # Functions

# %%
def convert_gene_pairs(gene_pairs, convert_to_entrezid=False):
    """
    Converts gene pair information (as dataframe) into a suitable format for the function process_tissue_networks.
    """
    gene_pairs = gene_pairs.reset_index()

    if convert_to_entrezid:
        gene_pairs = gene_pairs.replace(
            {
                "level_0": gene_symbol_to_entrezid,
                "level_1": gene_symbol_to_entrezid,
            }
        )

    gene_pairs = gene_pairs[["level_0", "level_1"]].itertuples(index=False, name=None)

    return list(gene_pairs)


# %%
def process_tissue_networks(gene_pairs, output_directory, force_tissue=None):
    """
    Given a list of tuples with gene pairs, it uses the GIANT web services to predict a
    relevant tissue for each gene pair and its gene network. Then it saves all the genes
    in the networks with their edges' values.

    If force_tissue is None, then autodetect the cell type for gene pairs.
    Otherwise, force_tissue should be a string, which will be used as key to query in
    dictionary TISSUE_SPECIFIC_URLS.
    """
    with tqdm(total=min(N_TOP_GENE_PAIRS, len(gene_pairs)), ncols=100) as pbar:
        gp_idx = 0

        while pbar.n < N_TOP_GENE_PAIRS and gp_idx < len(gene_pairs):
            gp = gene_pairs[gp_idx]

            pbar.set_description(",".join(gp))

            # check whether file already exists
            suffix = ""
            if force_tissue is not None:
                suffix = f"-{force_tissue}"

            output_filepath = (
                output_directory
                / f"{gp_idx:03d}-{gp[0].lower()}_{gp[1].lower()}{suffix}.h5"
            )
            if output_filepath.exists():
                gp_idx += 1
                pbar.update(1)
                continue

            output_directory.mkdir(exist_ok=True, parents=True)

            # predict a network for a gene pair
            _res = get_network(
                gene_symbols=gp,
                gene_ids_mappings=gene_id_mappings,
                tissue=TISSUE_SPECIFIC_URLS[force_tissue]
                if force_tissue is not None
                else None,
            )
            if _res is None:
                gp_idx += 1
                continue

            df, tissue, mincut = _res

            assert not df.isna().any().any()

            with pd.HDFStore(output_filepath, mode="w", complevel=4) as store:
                store.put("data", df, format="table")

                metadata = pd.DataFrame(
                    {
                        "tissue": tissue,
                        "mincut": mincut,
                    },
                    index=[0],
                )
                store.put("metadata", metadata, format="table")

            gp_idx += 1
            pbar.update(1)


# %% [markdown]
# # Predict tissue for each gene pair

# %% [markdown]
# ## Custom gene pairs from Figure 3

# %%
gene_pairs = [
    ("IFNG", "SDS"),
    ("JUN", "APOC1"),
]

display(len(gene_pairs))

# %% [markdown]
# ### Autodetected cell type

# %%
output_dir = OUTPUT_DIR / "custom" / "autopredicted_cell_type"

# %%
process_tissue_networks(gene_pairs, output_dir)

# %% [markdown]
# ### Blood

# %%
output_dir = OUTPUT_DIR / "custom" / "blood"

# %%
process_tissue_networks(
    gene_pairs
    + [
        ("ZDHHC12", "CCL18"),
        ("RASSF2", "CYTIP"),
        ("MYOZ1", "TNNI2"),
        ("PYGM", "TPM2"),
    ],
    output_dir,
    force_tissue="blood",
)

# %% [markdown]
# ## Clustermatch vs Pearson

# %%
output_dir = OUTPUT_DIR / "clustermatch_vs_pearson"

# %%
# read gene pairs
data = pd.read_pickle(INPUT_DIR / "clustermatch_vs_pearson.pkl").sort_values(
    "ccc", ascending=False
)

# %%
data.shape

# %%
data.head()

# %% [markdown]
# From the `data` dataframe, only gene pairs (index) are used.
# The other numbers are correlation values and their rankings.

# %%
gene_pairs = convert_gene_pairs(data)
display(len(gene_pairs))

# %%
gene_pairs[:10]

# %%
process_tissue_networks(gene_pairs, output_dir)

# %% [markdown]
# ## Clustermatch vs Pearson/Spearman

# %%
output_dir = OUTPUT_DIR / "clustermatch_vs_pearson_spearman"

# %%
data = pd.read_pickle(INPUT_DIR / "clustermatch_vs_pearson_spearman.pkl").sort_values(
    "ccc", ascending=False
)

# %%
data.shape

# %%
data.head()

# %%
gene_pairs = convert_gene_pairs(data)
display(len(gene_pairs))

# %%
gene_pairs[:10]

# %%
process_tissue_networks(gene_pairs, output_dir)

# %% [markdown]
# ## Clustermatch vs Spearman

# %%
output_dir = OUTPUT_DIR / "clustermatch_vs_spearman"

# %%
data = pd.read_pickle(INPUT_DIR / "clustermatch_vs_spearman.pkl").sort_values(
    "ccc", ascending=False
)

# %%
data.shape

# %%
data.head()

# %%
gene_pairs = convert_gene_pairs(data)
display(len(gene_pairs))

# %%
gene_pairs[:10]

# %%
process_tissue_networks(gene_pairs, output_dir)

# %% [markdown]
# ## Pearson vs Clustermatch

# %%
output_dir = OUTPUT_DIR / "pearson_vs_clustermatch"

# %%
data = pd.read_pickle(INPUT_DIR / "pearson_vs_clustermatch.pkl").sort_values(
    "pearson", ascending=False
)

# %%
data.shape

# %%
data.head()

# %%
gene_pairs = convert_gene_pairs(data)
display(len(gene_pairs))

# %%
gene_pairs[:10]

# %%
process_tissue_networks(gene_pairs, output_dir)

# %% [markdown]
# ## Pearson vs Clustermatch/Spearman

# %%
output_dir = OUTPUT_DIR / "pearson_vs_clustermatch_spearman"

# %%
data = pd.read_pickle(INPUT_DIR / "pearson_vs_clustermatch_spearman.pkl").sort_values(
    "pearson", ascending=False
)

# %%
data.shape

# %%
data.head()

# %%
gene_pairs = convert_gene_pairs(data)
display(len(gene_pairs))

# %%
gene_pairs[:10]

# %%
process_tissue_networks(gene_pairs, output_dir)

# %%
