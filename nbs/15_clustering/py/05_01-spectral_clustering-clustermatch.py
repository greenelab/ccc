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
#
# Old:
# This notebook runs some pre-analyses using spectral clustering to explore the best set of parameters to cluster `z_score_std` data version.

# %% [markdown] tags=[]
# # Environment variables

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

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
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
INITIAL_RANDOM_STATE = 12345

# %% tags=[]
METHOD_NAME = "clustermatch"

# %% tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_RANGE"] = [
    2,
    5,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75,
    80,
    90,
    95,
    100,
    200,
]
# CLUSTERING_OPTIONS["N_REPS_PER_K"] = 5
CLUSTERING_OPTIONS["KMEANS_N_INIT"] = 50
CLUSTERING_OPTIONS["DELTA"] = 0.25

display(CLUSTERING_OPTIONS)

# %% tags=[]
# BASE_FOLDER = Path("..", "base").resolve()
BASE_FOLDER = Path("base").resolve()
assert BASE_FOLDER.exists()

display(BASE_FOLDER)

# %% [markdown] tags=[]
# # Load Clustermatch correlations

# %% tags=[]
input_filepath = Path(
    BASE_FOLDER,
    "results",
    "sim_mat",
    "wb_data_gene_cm.pkl",
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
# ## Get distance matrix

# %% tags=[]
dist_matrix = 1.0 - sim_matrix

# %% [markdown] tags=[]
# # Clustering

# %% tags=[]
from clustering.methods import DeltaSpectralClustering
from sklearn.metrics import silhouette_score

# %% [markdown] tags=[]
# ## Clusterers

# %% tags=[]
np.sqrt(sim_matrix.shape[0])

# %% tags=[]
from clustering.methods import DeltaSpectralClustering

# %% tags=[]
CLUSTERERS = {}

idx = 0
random_state = INITIAL_RANDOM_STATE

for k in CLUSTERING_OPTIONS["K_RANGE"]:
    #     for delta_value in CLUSTERING_OPTIONS["DELTAS"]:
    #         for i in range(CLUSTERING_OPTIONS["N_REPS_PER_K"]):
    clus = DeltaSpectralClustering(
        eigen_solver="arpack",
        n_clusters=k,
        n_init=CLUSTERING_OPTIONS["KMEANS_N_INIT"],
        delta=CLUSTERING_OPTIONS["DELTA"],
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
import tempfile
from clustering.ensembles.utils import generate_ensemble
from utils import generate_result_set_name

# %% tags=[]
# generate a temporary folder where to store the ensemble and avoid computing it again
ensemble_folder = Path(
    BASE_FOLDER,
    "results",
    METHOD_NAME,
).resolve()
display(ensemble_folder)

ensemble_folder.mkdir(parents=True, exist_ok=True)

# %% tags=[]
ensemble_file = Path(
    ensemble_folder,
    generate_result_set_name(CLUSTERING_OPTIONS, prefix="ensemble-", suffix=".pkl"),
)
display(ensemble_file)

# %% tags=[]
if ensemble_file.exists():
    display("Ensemble file exists")
    ensemble = pd.read_pickle(ensemble_file)
else:
    ensemble = generate_ensemble(
        dist_matrix,
        CLUSTERERS,
        attributes=["n_clusters", "delta"],
    )

    ensemble["delta"] = ensemble["delta"].apply(lambda x: f"{x:.2f}")

# %% tags=[]
ensemble.shape

# %% tags=[]
ensemble.head()

# %% tags=[]
ensemble["n_clusters"].value_counts()

# %% tags=[]
_tmp = ensemble["n_clusters"].value_counts().unique()
assert _tmp.shape[0] == 1
assert _tmp[0] == 1

# %% tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] tags=[]
# ## Testing

# %% tags=[]
assert ensemble_stats["min"] > 1

# %% tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% tags=[]
# all partitions have the right size
assert np.all(
    [
        part["partition"].shape[0] == sim_matrix.shape[0]
        for idx, part in ensemble.iterrows()
    ]
)

# %% tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% tags=[]
# check that the number of clusters in the partitions are the expected ones
_real_k_values = ensemble["partition"].apply(lambda x: np.unique(x).shape[0])
display(_real_k_values)
assert np.all(ensemble["n_clusters"].values == _real_k_values.values)

# %% [markdown] tags=[]
# ## Add clustering quality measures

# %% tags=[]
ensemble = ensemble.assign(
    si_score=ensemble["partition"].apply(
        lambda x: silhouette_score(dist_matrix, x, metric="precomputed")
    ),
)

# %% tags=[]
ensemble.shape

# %% tags=[]
ensemble.head()

# %% [markdown] tags=[]
# ## Save

# %% tags=[]
ensemble.to_pickle(ensemble_file)

# %% [markdown] tags=[]
# # Cluster quality

# %% tags=[]
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    _df = ensemble.groupby(["n_clusters", "delta"]).mean()
    display(_df)

# %% tags=[]
with sns.plotting_context("talk", font_scale=0.75), sns.axes_style(
    "whitegrid", {"grid.linestyle": "--"}
):
    fig = plt.figure(figsize=(14, 6))
    ax = sns.pointplot(data=ensemble, x="n_clusters", y="si_score", hue="delta")
    ax.set_ylabel("Silhouette index\n(higher is better)")
    ax.set_xlabel("Number of clusters ($k$)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.grid(True)
    plt.tight_layout()

# %% tags=[]
