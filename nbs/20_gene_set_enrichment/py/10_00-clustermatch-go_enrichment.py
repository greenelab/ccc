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
METHOD = "clustermatch"
METHOD_SHORT = "cm"

# %% tags=[]
# BASE_FOLDER = Path("..", "base").resolve()
BASE_FOLDER = Path("base").resolve()

assert BASE_FOLDER.exists()

display(BASE_FOLDER)

# %% tags=[]
OUTPUT_DIR = Path(BASE_FOLDER, "results", METHOD, "enrichment").resolve()
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
    "ensemble-DELTA_025-KMEANS_N_INIT_50-K_RANGE_2_5_10_15_20_25_30_35_40_45_50_55_60_65_70_75_80_90_95_100_200.pkl",
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


# %% [markdown] tags=[]
# ## enrichGO

# %% tags=[]
# ck = clusterProfiler.enrichGO(
#     gene=genes_per_cluster[2],
#     OrgDb="org.Hs.eg.db",
#     keyType='ENSEMBL',
#     universe=genes_universe,
#     pAdjustMethod='fdr',
# #     pAdjustMethod='bonferroni',
#     pvalueCutoff=0.05,
# #     qvalueCutoff=0.05,
#     ont='BP',
#     readable=True,
# )

# %% tags=[]
# cks = clusterProfiler.simplify(ck, cutoff=0.3)
# cks = ck

# %% tags=[]
# list(cks.slots.keys())

# %% tags=[]
# df = cks.slots['result']

# %% tags=[]
# df.shape

# %% tags=[]
# df.sort_values('p.adjust').head()

# %% [markdown] tags=[]
# ## enrichPathway

# %% tags=[]
# genes_universe_entrez = clusterProfiler.bitr(
#     genes_universe, fromType="ENSEMBL", toType="ENTREZID", OrgDb="org.Hs.eg.db"
# ).drop_duplicates(subset=['ENSEMBL'])['ENTREZID'].tolist()

# %% tags=[]
# genes_per_cluster_set_entrez = {
#     f'C{k:n}': clusterProfiler.bitr(v, fromType="ENSEMBL", toType="ENTREZID", OrgDb="org.Hs.eg.db").drop_duplicates(subset=['ENSEMBL'])['ENTREZID'].tolist()
#     for k, v in genes_per_cluster.items()
# }

# genes_per_cluster_set_entrez = robjects.ListVector(genes_per_cluster_set_entrez)

# %% tags=[]
# ck = reactomePA.enrichPathway(
#     gene=genes_per_cluster_set_entrez[0],
#     universe=genes_universe_entrez,
#     pAdjustMethod='fdr',
# #     pAdjustMethod='bonferroni',
#     pvalueCutoff=0.05,
# #     qvalueCutoff=0.05,
#     readable=True,
# )

# %% tags=[]
# cks = clusterProfiler.simplify(ck, cutoff=0.3)
# cks = ck

# %% tags=[]
# list(cks.slots.keys())

# %% tags=[]
# df = cks.slots['result']

# %% tags=[]
# df.shape

# %% tags=[]
# df.sort_values('p.adjust').head()

# %% [markdown] tags=[]
# ## compareClusters

# %% tags=[]
def run_enrich(filename_prefix, partition, ontology):
    display(f"  {ontology}")

    genes_per_cluster = {}
    for c in pd.Series(partition).value_counts().index:
        genes_per_cluster[c] = [
            g.split(".")[0] for g in sim_matrix.index[partition == c]
        ]

    genes_per_cluster_set = {
        f"C{k:n}": list(set(v)) for k, v in genes_per_cluster.items()
    }
    gene_clusters = robjects.ListVector(genes_per_cluster_set)

    ck = clusterProfiler.compareCluster(
        geneClusters=gene_clusters,
        OrgDb="org.Hs.eg.db",
        keyType="ENSEMBL",
        universe=genes_universe,
        fun="enrichGO",
        pAdjustMethod="fdr",
        pvalueCutoff=0.05,
        ont=ontology,
        readable=True,
    )

    df = ck.slots["compareClusterResult"]
    display(f"    Full results size: {df.shape}")
    df.sort_values("p.adjust").to_pickle(
        Path(
            OUTPUT_DIR,
            f"{filename_prefix}{METHOD}-enrichGO-{ontology}-full.pkl",
        )
    )

    # simplify
    ck = clusterProfiler.simplify(ck, cutoff=0.3)
    df = ck.slots["compareClusterResult"]
    display(f"    Simplified results size: {df.shape}")
    df.sort_values("p.adjust").to_pickle(
        Path(
            OUTPUT_DIR,
            f"{filename_prefix}{METHOD}-enrichGO-{ontology}-simplified.pkl",
        )
    )


# %% tags=[]
# ck = clusterProfiler.compareCluster(
#     geneClusters=genes_per_cluster_set_entrez,
#     universe=genes_universe_entrez,
# #     fun='enrichGO',
#     fun='enrichPathway',
#     pAdjustMethod='fdr',
# #     pAdjustMethod='bonferroni',
#     pvalueCutoff=0.05,
#     readable=True,
# )

# %% tags=[]
for idx, cr in cm_results.sort_values("n_clusters").iterrows():
    display(f"Partition with n_clusters={cr.n_clusters}")

    prefix = f"k_{cr.n_clusters}-"
    run_enrich(prefix, cr.partition, "BP")
    run_enrich(prefix, cr.partition, "CC")
    run_enrich(prefix, cr.partition, "MF")

# %% [markdown] tags=[]
# ## Plot

# %% tags=[]
# import rpy2.robjects.lib.ggplot2 as ggplot2

# %% tags=[]
# grdevices.pdf(file=os.path.join(OUTPUT_FIGURES_DIR, 'asthma_biclusters_go_enrichment.pdf'), width=13, height=8)
# args = {'showCategory': 5, 'font.size': 12}
# p = enrichplot.dotplot(cks, **args) + \
#     ggplot2.theme(**{
#         'text': ggplot2.element_text(size=20),
#         'axis.text.x': ggplot2.element_text(size = 19),
#         'axis.text.y': ggplot2.element_text(angle = 0, hjust = 1, size = 18),
#         'legend.title.align': 0.5,
#     }) + \
#     ggplot2.labs(color = "p-value\n(FDR)", size = 'Gene\nratio')
# rprint(p)
# grdevices.dev_off()

# %% tags=[]
