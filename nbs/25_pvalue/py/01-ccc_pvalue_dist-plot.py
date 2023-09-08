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
# Generates plots of the CCC pvalues from the null distribution.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from ccc import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
OUTPUT_DIR = conf.RESULTS_DIR / "ccc_null-pvalues"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% tags=[]
OUTPUT_DIR

# %% [markdown] tags=[]
# # Load CCC values and pvalues

# %% tags=[]
output_file = OUTPUT_DIR / "cm_values.npy"
cm_values = np.load(output_file)
display(cm_values.shape)

# %% tags=[]
output_file = OUTPUT_DIR / "cm_pvalues.npy"
cm_pvalues = np.load(output_file)
display(cm_pvalues.shape)

# %% tags=[]
n_perms = cm_pvalues.shape[0]
min_pvalue_resolution = (0 + 1) / (n_perms + 1)
display(min_pvalue_resolution)

# %% [markdown] tags=[]
# # Plots

# %% tags=[]
plt.hist(cm_pvalues, bins=10, edgecolor="k")  # Adjust the number of bins as needed
plt.title("Distribution of Values")
plt.xlabel("Value")
plt.ylabel("Frequency")

# %% tags=[]
sns.histplot(cm_pvalues, kde=True, color="blue")
plt.title("Distribution of Values")
plt.xlabel("Value")
plt.ylabel("Density")

# %% [markdown] tags=[]
# # KS

# %% tags=[]
stats.kstest(
    cm_pvalues,
    stats.uniform.cdf,
    args=(min_pvalue_resolution, 1 - min_pvalue_resolution),
)

# %% tags=[]
