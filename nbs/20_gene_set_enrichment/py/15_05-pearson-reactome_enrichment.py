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
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path

import pandas as pd

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
METHOD = "pearson"
METHOD_SHORT = "corr"

# %% tags=[]
# BASE_FOLDER = Path("..", "base").resolve()
BASE_FOLDER = Path("base").resolve()

assert BASE_FOLDER.exists()

display(BASE_FOLDER)

# %% tags=[]
OUTPUT_DIR = Path(BASE_FOLDER, "results", METHOD, "enrichPathway").resolve()
display(OUTPUT_DIR)

OUTPUT_DIR.mkdir(exist_ok=True)

# %% [markdown] tags=[]
# # Load correlations

# %% tags=[]
input_filepath = Path(
    BASE_FOLDER,
    "results",
    "sim_mat",
    f"wb_data_gene_{METHOD_SHORT}.pkl",
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% tags=[]
sim_matrix = pd.read_pickle(input_filepath)

# %% tags=[]
sim_matrix.shape

# %% tags=[]
sim_matrix.head()

# %% [markdown] tags=[]
# # Load clustering results

# %% tags=[]
ensemble_folder = Path(
    BASE_FOLDER,
    "results",
).resolve()
display(ensemble_folder)

# ensemble_folder.mkdir(parents=True, exist_ok=True)

# %% tags=[]
input_filepath = Path(
    ensemble_folder,
    METHOD,
    "ensemble-DELTA_035-KMEANS_N_INIT_50-K_RANGE_2_5_10_15_20_25_30_35_40_45_50_55_60_65_70_75_80_90_95_100_200.pkl",
).resolve()
display(input_filepath)

# %% tags=[]
cm_results = pd.read_pickle(input_filepath)

# %% tags=[]
cm_results.shape

# %% tags=[]
cm_results.head()

# %% tags=[]
cm_results["n_clusters"].unique()

# %% [markdown] tags=[]
# # clusterProfiler

# %% tags=[]
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

pandas2ri.activate()

# %% tags=[]
rprint = robjects.globalenv.find("print")
# dose = importr('DOSE')
clusterProfiler = importr("clusterProfiler")
reactomePA = importr("ReactomePA")
# enrichplot = importr("enrichplot")
grdevices = importr("grDevices")

# %% [markdown] tags=[]
# ## General variables

# %% tags=[]
genes_universe = [g.split(".")[0] for g in sim_matrix.index]

# %% tags=[]
len(genes_universe)

# %% tags=[]
# Convert from ENSEMBL TO ENTREZID
genes_universe = (
    clusterProfiler.bitr(
        genes_universe, fromType="ENSEMBL", toType="ENTREZID", OrgDb="org.Hs.eg.db"
    )
    .drop_duplicates(subset=["ENSEMBL"])["ENTREZID"]
    .tolist()
)

# %% [markdown] tags=[]
# ## compareClusters

# %% tags=[]
import warnings


# %% tags=[]
def run_enrich(filename_prefix, partition):
    genes_per_cluster = {}
    for c in pd.Series(partition).value_counts().index:
        genes_per_cluster[c] = [
            g.split(".")[0] for g in sim_matrix.index[partition == c]
        ]

    #     genes_per_cluster_set = {
    #         f"C{k:n}": list(set(v)) for k, v in genes_per_cluster.items()
    #     }
    #     gene_clusters = robjects.ListVector(genes_per_cluster_set)

    try:
        genes_per_cluster_set = {
            f"C{k:n}": clusterProfiler.bitr(
                v, fromType="ENSEMBL", toType="ENTREZID", OrgDb="org.Hs.eg.db"
            )
            .drop_duplicates(subset=["ENSEMBL"])["ENTREZID"]
            .tolist()
            for k, v in genes_per_cluster.items()
        }
        gene_clusters = robjects.ListVector(genes_per_cluster_set)
    except:
        warnings.warn(f"Partitions {filename_prefix} failed.")
        return

    ck = clusterProfiler.compareCluster(
        geneClusters=gene_clusters,
        universe=genes_universe,
        organism="human",
        fun="enrichPathway",
        pAdjustMethod="fdr",
        pvalueCutoff=0.05,
        readable=True,
    )

    df = ck.slots["compareClusterResult"]
    display(f"    Full results size: {df.shape}")
    df.sort_values("p.adjust").to_pickle(
        Path(
            OUTPUT_DIR,
            f"{filename_prefix}{METHOD}-enrichPathway-full.pkl",
        )
    )


# %% tags=[]
for idx, cr in cm_results.sort_values("n_clusters").iterrows():
    display(f"Partition with n_clusters={cr.n_clusters}")

    prefix = f"k_{cr.n_clusters}-"
    run_enrich(prefix, cr.partition)

# %% tags=[]
