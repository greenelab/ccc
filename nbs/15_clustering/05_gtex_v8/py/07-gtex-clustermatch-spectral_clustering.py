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
# # Environment variables

# %% tags=[]
# # %load_ext autoreload
# # %autoreload 2

# %% tags=[]
from IPython.display import display

N_JOBS = 2

# %% tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns

from clustermatch import conf
from clustermatch.utils import simplify_string

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
CORRELATION_METHOD_NAME = "clustermatch"

# %% tags=[]
GENE_SELECTION_STRATEGY = "var_pc_log2"

# %%
# Tissues with largest sample size from GTEx (see nbs/05_preprocessing/00-gtex_v8-split_by_tissue.ipynb)
TISSUES = [
    "Muscle - Skeletal",
    "Whole Blood",
    "Skin - Sun Exposed (Lower leg)",
    "Adipose - Subcutaneous",
    "Artery - Tibial",
]

# %%
K_RANGE = [2] + np.arange(5, 100 + 1, 5).tolist() + [125, 150, 175, 200]

# %%
N_INIT = 50

# %% tags=[]
INITIAL_RANDOM_STATE = 12345


# %%
def process_similarity_matrix(similarity_matrix):
    # for clustermatch, negative values are meaningless, so we replace them by zero
    similarity_matrix[similarity_matrix < 0.0] = 0.0
    return similarity_matrix


# %%
def get_distance_matrix(similarity_matrix):
    """
    Converts the processed similarity matrix into a distance matrix.
    """
    return 1.0 - similarity_matrix


# %% [markdown]
# # Paths

# %%
INPUT_DIR = conf.GTEX["SIMILARITY_MATRICES_DIR"]
display(INPUT_DIR)
assert INPUT_DIR.exists()

# %%
OUTPUT_DIR = conf.GTEX["CLUSTERING_DIR"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(INPUT_DIR)

# %% [markdown]
# # Setup clustering options

# %% tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_RANGE"] = K_RANGE
CLUSTERING_OPTIONS["KMEANS_N_INIT"] = N_INIT

display(CLUSTERING_OPTIONS)

# %% [markdown] tags=[]
# # Get data files

# %%
tissue_names = [simplify_string(t.lower()) for t in TISSUES]
display(tissue_names)

# %% tags=[]
input_files = sorted(list(INPUT_DIR.glob(f"*-{GENE_SELECTION_STRATEGY}-{CORRELATION_METHOD_NAME}.pkl")))
input_files = [
    f for f in input_files if any(f"gtex_v8_data_{tn}-" in f.name for tn in tissue_names)
]
display(len(input_files))
display(input_files)

assert len(input_files) == len(TISSUES), len(TISSUES)
display(input_files[:5])

# %% [markdown]
# ## Show the content of one similarity matrix

# %% tags=[]
sim_matrix = pd.read_pickle(input_files[0])

# %% tags=[]
sim_matrix.shape

# %% tags=[]
sim_matrix.head()

# %% [markdown] tags=[]
# # Clustering

# %% tags=[]
# from clustering.methods import DeltaSpectralClustering
# from sklearn.metrics import silhouette_score

# %% [markdown] tags=[]
# ## Clusterers

# %% tags=[]
# np.sqrt(sim_matrix.shape[0])

# %% tags=[]
# from clustering.methods import DeltaSpectralClustering

# %% tags=[]
CLUSTERERS = {}

idx = 0
random_state = INITIAL_RANDOM_STATE

for k in CLUSTERING_OPTIONS["K_RANGE"]:
    #     for delta_value in CLUSTERING_OPTIONS["DELTAS"]:
    #         for i in range(CLUSTERING_OPTIONS["N_REPS_PER_K"]):
    clus = SpectralClustering(
        eigen_solver="arpack",
        n_clusters=k,
        n_init=CLUSTERING_OPTIONS["KMEANS_N_INIT"],
        affinity="precomputed",
        random_state=random_state,
    )

    method_name = type(clus).__name__
    CLUSTERERS[f"{method_name} #{idx}"] = clus

    random_state = random_state + 1
    idx = idx + 1

# %% tags=[]
display(len(CLUSTERERS))

# %% tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] tags=[]
# ## Generate ensemble

# %% tags=[]
from sklearn.metrics import silhouette_score
from clustermatch.clustering import generate_ensemble
# from utils import generate_result_set_name

# %% tags=[]
# # generate a temporary folder where to store the ensemble and avoid computing it again
# ensemble_folder = Path(
#     BASE_FOLDER,
#     "results",
#     METHOD_NAME,
# ).resolve()
# display(ensemble_folder)

# ensemble_folder.mkdir(parents=True, exist_ok=True)

# %%
pbar = tqdm(input_files, ncols=100)

for tissue_data_file in pbar:
    pbar.set_description(tissue_data_file.stem)

    # read
    sim_matrix = pd.read_pickle(tissue_data_file)
    sim_matrix = process_similarity_matrix(sim_matrix)

    ensemble = generate_ensemble(
        sim_matrix,
        CLUSTERERS,
        attributes=["n_clusters"],
        tqdm_args={"leave": False, "ncols": 100},
    )
    
    _tmp = ensemble["n_clusters"].value_counts().unique()
    assert _tmp.shape[0] == 1
    assert _tmp[0] == 1
    
    assert not ensemble["n_clusters"].isna().any()
    
    assert ensemble.shape[0] == len(CLUSTERERS)
    
    assert np.all(
        [
            part["partition"].shape[0] == sim_matrix.shape[0]
            for idx, part in ensemble.iterrows()
        ]
    )
    
    # no partition has negative labels or nan
    assert not np.any([np.isnan(part["partition"]).any() for idx, part in ensemble.iterrows()])
    assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])
    
    _real_k_values = ensemble["partition"].apply(lambda x: np.unique(x).shape[0])
    display(_real_k_values)
    assert np.all(ensemble["n_clusters"].values == _real_k_values.values)
    
    # add clustering quality measures
    dist_matrix = get_distance_matrix(sim_matrix)
    ensemble = ensemble.assign(
        si_score=ensemble["partition"].apply(
            lambda x: silhouette_score(dist_matrix, x, metric="precomputed")
        ),
    )

    # save
    output_filename = f"{tissue_data_file.stem}-{CORRELATION_METHOD_NAME}.pkl"
    ensemble.to_pickle(path=OUTPUT_DIR / output_filename)

# %% [markdown] tags=[]
# # Cluster quality

# %% [markdown] tags=[]
# **TODO**: move this to another notebook

# %% tags=[]
# with pd.option_context("display.max_rows", None, "display.max_columns", None):
#     _df = ensemble.groupby(["n_clusters", "delta"]).mean()
#     display(_df)

# %% tags=[]
# with sns.plotting_context("talk", font_scale=0.75), sns.axes_style(
#     "whitegrid", {"grid.linestyle": "--"}
# ):
#     fig = plt.figure(figsize=(14, 6))
#     ax = sns.pointplot(data=ensemble, x="n_clusters", y="si_score", hue="delta")
#     ax.set_ylabel("Silhouette index\n(higher is better)")
#     ax.set_xlabel("Number of clusters ($k$)")
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
#     plt.grid(True)
#     plt.tight_layout()

# %% tags=[]
