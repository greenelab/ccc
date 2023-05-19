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

# %% [markdown]
# # Description

# %% [markdown]
# This notebook downloads the expression dataset described in:
# * https://pubmed.ncbi.nlm.nih.gov/27655842/
# * https://doi.org/10.1093/bib/bbab495

# %% [markdown]
# # Define magic

# %%
from IPython import get_ipython
from IPython.core.magic import register_cell_magic

ipython = get_ipython()


@register_cell_magic
def pybash(line, cell):
    ipython.run_cell_magic('bash', '', cell.format(**globals()))


# %% [markdown]
# # Modules

# %%
from ccc import conf

# %% [markdown]
# # Install pysradb

# %%
conda_env = conf.CONDA_ENVS_DIR / "pysradb"
conda_env.parent.mkdir(parents=True, exist_ok=True)
display(conda_env)

# %%
# %%pybash
. ~/.bashrc
conda create -y -p {conda_env} -c bioconda python=3.10.* pysradb=2.0.*

# %%
# %%pybash
. ~/.bashrc
conda activate {conda_env}
python --version

# %%
# %%pybash
. ~/.bashrc
conda activate {conda_env}

pysradb --version

# %% [markdown]
# # Download processed gene expression data

# %%
download_dir = conf.DATA_DIR / "rice"
download_dir.mkdir(parents=True, exist_ok=True)
display(download_dir)

# %%
# %%pybash
. ~/.bashrc
conda activate {conda_env}

# download
pysradb download -g GSE74793 --out-dir {download_dir}

# rename file
# mv {download_dir}/GSE74793/GSE74793_processed-data.txt.gz {download_dir}/gene_expr_processed-data.txt.gz
# rm -rf {download_dir}/GSE74793/

# %% [markdown]
# # Download metadata

# %%
# %%pybash
{conda_env}/bin/pysradb metadata PRJNA301554 --detailed --saveto {download_dir}/metadata.tsv

# %%
