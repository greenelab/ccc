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
import re

import pandas as pd

# from scipy import stats
# import seaborn as sns

# from clustermatch.plots import plot_histogram, plot_cumulative_histogram, jointplot
from clustermatch import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
# DATASET_CONFIG = conf.GTEX
# GTEX_TISSUE = "whole_blood"
# GENE_SEL_STRATEGY = "var_pc_log2"

# %% tags=[]
# # this is used for the cumulative histogram
# GENE_PAIRS_PERCENT = 0.70

# %%
# CLUSTERMATCH_LABEL = "Clustermatch"
# PEARSON_LABEL = "Pearson"
# SPEARMAN_LABEL = "Spearman"

# %%
GENE_FILE_MARK_TEMPLATE = "| *{gene}* |"

# %%
GENE0_STATS_TEMPLATE = '| *{gene}* | ADD DIRECT<!-- $rowspan="2" --> | {blood_min} | {blood_avg} | {blood_max} | ADD DIRECT<!-- $rowspan="2" --> | {pred_min} | {pred_avg} | {pred_max} |'
GENE1_STATS_TEMPLATE = '| *{gene}* | {blood_min} | {blood_avg} | {blood_max} | {pred_min} | {pred_avg} | {pred_max}<!-- $removenext="3" --> |'

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
# assert (
#     conf.MANUSCRIPT["BASE_DIR"] is not None and conf.MANUSCRIPT["BASE_DIR"].exists()
# ), "Manuscript dir not set"

# %% tags=[]
# OUTPUT_FIGURE_DIR = (
#     conf.MANUSCRIPT["FIGURES_DIR"] / "coefs_comp" / f"gtex_{GTEX_TISSUE}"
# )
# OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
# display(OUTPUT_FIGURE_DIR)

# %% tags=[]
# INPUT_CORR_FILE_TEMPLATE = (
#     DATASET_CONFIG["SIMILARITY_MATRICES_DIR"]
#     / DATASET_CONFIG["SIMILARITY_MATRIX_FILENAME_TEMPLATE"]
# )
# display(INPUT_CORR_FILE_TEMPLATE)

# %% tags=[]
# INPUT_FILE = DATASET_CONFIG["SIMILARITY_MATRICES_DIR"] / str(
#     INPUT_CORR_FILE_TEMPLATE
# ).format(
#     tissue=GTEX_TISSUE,
#     gene_sel_strategy=GENE_SEL_STRATEGY,
#     corr_method="all",
# )
# display(INPUT_FILE)

# assert INPUT_FILE.exists()

# %% tags=[]
INPUT_DIR = conf.GIANT["RESULTS_DIR"]
display(INPUT_DIR)

assert INPUT_DIR.exists()


# %% [markdown] tags=[]
# # Functions

# %%
def read_data(gene0, gene1, tissue_name, mode="visible"):
    input_filename = f"{gene0.lower()}_{gene1.lower()}-network-{mode}.csv"

    input_filepath = INPUT_DIR / tissue_name / input_filename
    assert input_filepath.exists()

    data = pd.read_csv(input_filepath)

    assert (
        (gene0 in data["GENE1"].unique()) or (gene0 in data["GENE2"].unique())
    ) and ((gene1 in data["GENE1"].unique()) or (gene1 in data["GENE2"].unique()))

    return data


# %%
# testing
_tmp0 = read_data("IFNG", "SDS", "blood")
display(_tmp0.shape)

_tmp1 = read_data("IFNG", "SDS", "pred")
display(_tmp1.shape)

assert _tmp0.shape[0] != _tmp1.shape[0]


# %%
def format_number(number):
    return f"{number:.2f}"


# %%
# testing
format_number(0.222222) == "0.22"
format_number(0.225222) == "0.23"


# %%
def get_gene_stats(df, gene_name):
    gene_data = df[(df["GENE1"] == gene_name) | (df["GENE2"] == gene_name)]
    return gene_data.describe().squeeze()


# %%
# testing
_tmp0_stats = get_gene_stats(_tmp0, "IFNG")
assert _tmp0_stats["min"].round(2) == 0.19
assert _tmp0_stats["mean"].round(2) == 0.42
assert _tmp0_stats["max"].round(2) == 0.54


# %%
def get_gene_content(blood_stats, pred_stats, gene_name, gene_template):
    return gene_template.format(
        gene=gene_name,
        blood_min=format_number(blood_stats["min"]),
        blood_avg=format_number(blood_stats["mean"]),
        blood_max=format_number(blood_stats["max"]),
        pred_min=format_number(pred_stats["min"]),
        pred_avg=format_number(pred_stats["mean"]),
        pred_max=format_number(pred_stats["max"]),
    )


# %%
# testing
_tmp_gene_cont = get_gene_content(
    _tmp0_stats, _tmp0_stats, "IFNG", GENE0_STATS_TEMPLATE
)
assert "IFNG" in _tmp_gene_cont
assert "0.19" in _tmp_gene_cont
assert "0.42" in _tmp_gene_cont
assert "0.54" in _tmp_gene_cont


# %%
def write_content(text, text_replacement):
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
def process_genes(gene0, gene1):
    data_blood = read_data(gene0, gene1, "blood")
    data_pred = read_data(gene0, gene1, "pred")

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
            get_gene_content(blood_stats, pred_stats, gene_name, gene_template) + "\n"
        )

        gene_file_mark = GENE_FILE_MARK_TEMPLATE.format(gene=gene_name)

        write_content(gene_file_mark, new_content)


# %% [markdown] tags=[]
# # IFNG - SDS

# %%
process_genes("IFNG", "SDS")

# %% [markdown] tags=[]
# # JUN - APOC1

# %%
process_genes("JUN", "APOC1")

# %% [markdown] tags=[]
# # ZDHHC12 - CCL18

# %%
process_genes("ZDHHC12", "CCL18")

# %% [markdown] tags=[]
# # RASSF2 - CYTIP

# %%
process_genes("RASSF2", "CYTIP")

# %% [markdown] tags=[]
# # MYOZ1 - TNNI2

# %%
process_genes("MYOZ1", "TNNI2")

# %% [markdown] tags=[]
# # PYGM - TPM2

# %%
process_genes("PYGM", "TPM2")

# %%
