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
# For some gene pairs of interest, it reads the probabilities of interactions in predicted networks from GIANT.
# Then it writes networks stats in a table in a markdown file (from the manuscript).
# Two networks per gene pair are read/written: blood and an autodetected cell type (from GIANT).

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import re
from functools import partial

import pandas as pd

from clustermatch import conf

# %% [markdown] tags=[]
# # Settings

# %%
GENE_FILE_MARK_TEMPLATE = "| *{gene}* |"

# %%
GENE0_STATS_TEMPLATE = '| *{gene}* | {blood_min} | {blood_avg} | {blood_max} | {cell_type}<!-- $rowspan="2" --> | {pred_min} | {pred_avg} | {pred_max} |'
GENE1_STATS_TEMPLATE = '| *{gene}* | {blood_min} | {blood_avg} | {blood_max} | {pred_min} | {pred_avg} | {pred_max}<!-- $removenext="2" --> |'

# %% [markdown] tags=[]
# # Paths

# %%
assert (
    conf.MANUSCRIPT["BASE_DIR"] is not None
), "The manuscript directory was not configured"

# %%
OUTPUT_FILE_PATH = conf.MANUSCRIPT["CONTENT_DIR"] / "20.00.supplementary_material.md"
display(OUTPUT_FILE_PATH)
assert OUTPUT_FILE_PATH.exists()

# %% tags=[]
INPUT_DIR = conf.GIANT["RESULTS_DIR"] / "intersection_genes"
display(INPUT_DIR)

assert INPUT_DIR.exists()


# %% [markdown] tags=[]
# # Functions

# %%
def read_data(gene0, gene1, tissue_name=None, return_predicted_tissue=False):
    """
    Given a pair of genes, it returns the GIANT network data.
    If tissue_name is not None, it specifies the name of the tissue.
    If None, it means the autodetected tissue/cell type.
    """
    tissue_suffix = f"-{tissue_name}" if tissue_name is not None else ""

    file_pattern = f"???-{gene0.lower()}_{gene1.lower()}{tissue_suffix}.h5"
    files = list(INPUT_DIR.rglob(file_pattern))
    if len(files) == 0:
        file_pattern = f"???-{gene1.lower()}_{gene0.lower()}{tissue_suffix}.h5"
        files = list(INPUT_DIR.rglob(file_pattern))

    assert len(files) == 1
    input_filepath = files[0]
    assert input_filepath.exists()

    data = pd.read_hdf(input_filepath, key="data")

    assert (
        (gene0 in data["gene1"].unique()) or (gene0 in data["gene2"].unique())
    ) and ((gene1 in data["gene1"].unique()) or (gene1 in data["gene2"].unique()))

    if return_predicted_tissue:
        return data, pd.read_hdf(input_filepath, key="metadata").iloc[0]["tissue"]

    return data


# %%
# testing
_tmp0 = read_data("IFNG", "SDS", "blood")
assert _tmp0.shape[0] == 127
display(_tmp0.shape)

_tmp1 = read_data("IFNG", "SDS")
assert _tmp1.shape[0] == 124
display(_tmp1.shape)

_tmp1_tissue = read_data("IFNG", "SDS", return_predicted_tissue=True)[1]
assert _tmp1_tissue == "natural-killer-cell"

_tmp10 = read_data("ZDHHC12", "CCL18")
assert _tmp10.shape[0] > 1
_tmp11 = read_data("CCL18", "ZDHHC12")
assert _tmp11.shape == _tmp10.shape


# %%
def format_number(number):
    return f"{number:.2f}"


# %%
# testing
assert format_number(0.222222) == "0.22"
assert format_number(0.225222) == "0.23"


# %%
def get_gene_stats(df, gene_name):
    """
    Returns stats of interaction probabilities for a gene in data.
    """
    gene_data = df[(df["gene1"] == gene_name) | (df["gene2"] == gene_name)]
    return gene_data.describe().squeeze()


# %%
# testing
_tmp0_stats = get_gene_stats(_tmp0, "IFNG")
assert _tmp0_stats["min"].round(2) == 0.19
assert _tmp0_stats["mean"].round(2) == 0.42
assert _tmp0_stats["max"].round(2) == 0.54


# %%
def get_gene_content(blood_stats, pred_stats, gene_name, gene_template, cell_type=None):
    """
    Returns a string (from a template) with the data fields filled in.
    """
    s = partial(
        gene_template.format,
        gene=gene_name,
        blood_min=format_number(blood_stats["min"]),
        blood_avg=format_number(blood_stats["mean"]),
        blood_max=format_number(blood_stats["max"]),
        pred_min=format_number(pred_stats["min"]),
        pred_avg=format_number(pred_stats["mean"]),
        pred_max=format_number(pred_stats["max"]),
    )

    if "{cell_type}" in gene_template and cell_type is not None:
        return s(cell_type=cell_type)

    return s()


# %%
# testing
_tmp_gene_cont = get_gene_content(
    _tmp0_stats, _tmp0_stats, "IFNG", GENE0_STATS_TEMPLATE, "blood"
)
assert "IFNG" in _tmp_gene_cont
assert "0.19" in _tmp_gene_cont
assert "0.42" in _tmp_gene_cont
assert "0.54" in _tmp_gene_cont
assert "blood" in _tmp_gene_cont

# %%
# testing
_tmp_gene_cont = get_gene_content(
    _tmp0_stats, _tmp0_stats, "IFNG", GENE1_STATS_TEMPLATE
)
assert "IFNG" in _tmp_gene_cont
assert "0.19" in _tmp_gene_cont
assert "0.42" in _tmp_gene_cont
assert "0.54" in _tmp_gene_cont

# %%
# testing
_tmp_gene_cont = get_gene_content(
    _tmp0_stats, _tmp0_stats, "IFNG", GENE1_STATS_TEMPLATE, "blood"
)
assert "IFNG" in _tmp_gene_cont
assert "0.19" in _tmp_gene_cont
assert "0.42" in _tmp_gene_cont
assert "0.54" in _tmp_gene_cont
assert "blood" not in _tmp_gene_cont


# %%
def write_content(text, text_replacement):
    """
    It writes the table content in the output file.
    """
    with open(OUTPUT_FILE_PATH, "r", encoding="utf8") as f:
        file_content = f.read()

    new_file_content = re.sub(
        re.escape(text) + ".+\n",
        text_replacement,
        file_content,
        # flags=re.DOTALL,
    )

    with open(OUTPUT_FILE_PATH, "w", encoding="utf8") as f:
        f.write(new_file_content)


# %%
def format_tissue_name(tissue_name):
    s = " ".join(tissue_name.split("-"))
    s = list(s)
    s[0] = s[0].upper()
    return "".join(s)


# %%
# testing
assert format_tissue_name("blood") == "Blood"
assert format_tissue_name("natural-killer-cell") == "Natural killer cell"


# %%
def process_genes(gene0, gene1):
    """
    Given a gene pair, it updates a table in a Markdown file with statistics on their network data (GIANT),
    (such as network connectivity stats).
    """
    data_blood = read_data(gene0, gene1, "blood")
    data_pred, pred_tissue = read_data(gene0, gene1, return_predicted_tissue=True)

    for gene_name, gene_template in (
        (gene0, GENE0_STATS_TEMPLATE),
        (gene1, GENE1_STATS_TEMPLATE),
    ):
        blood_stats = get_gene_stats(data_blood, gene_name).rename(
            f"{gene_name} - blood"
        )
        display(blood_stats)

        pred_stats = get_gene_stats(data_pred, gene_name).rename(f"{gene_name} - pred")
        display(pred_stats)

        new_content = (
            get_gene_content(
                blood_stats,
                pred_stats,
                gene_name,
                gene_template,
                format_tissue_name(pred_tissue),
            )
            + "\n"
        )

        gene_file_mark = GENE_FILE_MARK_TEMPLATE.format(gene=gene_name)

        write_content(gene_file_mark, new_content)


# %% [markdown] tags=[]
# # Run

# %% [markdown] tags=[]
# Here I update the table for some gene pairs of interest in the manuscript.

# %% [markdown] tags=[]
# ## IFNG - SDS

# %%
process_genes("IFNG", "SDS")

# %% [markdown] tags=[]
# ## JUN - APOC1

# %%
process_genes("JUN", "APOC1")

# %% [markdown] tags=[]
# ## ZDHHC12 - CCL18

# %%
process_genes("ZDHHC12", "CCL18")

# %% [markdown] tags=[]
# ## RASSF2 - CYTIP

# %%
process_genes("RASSF2", "CYTIP")

# %% [markdown] tags=[]
# ## MYOZ1 - TNNI2

# %%
process_genes("MYOZ1", "TNNI2")

# %% [markdown] tags=[]
# ## PYGM - TPM2

# %%
process_genes("PYGM", "TPM2")

# %%
