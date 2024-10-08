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
# It generates different general plots to compare coefficient values from Pearson, Spearman and CCC, such as their distribution.
#
# In `Settings` below, the data set and other options (such as tissue for GTEx) are specified.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import pandas as pd
from scipy import stats
import seaborn as sns

from ccc.plots import plot_histogram, plot_cumulative_histogram, jointplot
from ccc import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
DATASET_CONFIG = conf.GTEX
GTEX_TISSUE = "whole_blood"
GENE_SEL_STRATEGY = "var_pc_log2"

# %% tags=[]
# this is used for the cumulative histogram
GENE_PAIRS_PERCENT = 0.70

# %% tags=[]
CLUSTERMATCH_LABEL = "CCC"
PEARSON_LABEL = "Pearson"
SPEARMAN_LABEL = "Spearman"

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
assert (
    conf.MANUSCRIPT["BASE_DIR"] is not None and conf.MANUSCRIPT["BASE_DIR"].exists()
), "Manuscript dir not set"

# %% tags=[]
OUTPUT_FIGURE_DIR = (
    conf.MANUSCRIPT["FIGURES_DIR"] / "coefs_comp" / f"gtex_{GTEX_TISSUE}"
)
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_FIGURE_DIR)

# %% tags=[]
INPUT_CORR_FILE_TEMPLATE = (
    DATASET_CONFIG["SIMILARITY_MATRICES_DIR"]
    / DATASET_CONFIG["SIMILARITY_MATRIX_FILENAME_TEMPLATE"]
)
display(INPUT_CORR_FILE_TEMPLATE)

# %% tags=[]
INPUT_FILE = DATASET_CONFIG["SIMILARITY_MATRICES_DIR"] / str(
    INPUT_CORR_FILE_TEMPLATE
).format(
    tissue=GTEX_TISSUE,
    gene_sel_strategy=GENE_SEL_STRATEGY,
    corr_method="all",
)
display(INPUT_FILE)

assert INPUT_FILE.exists()

# %% [markdown] tags=[]
# # Data

# %% tags=[]
df = pd.read_pickle(INPUT_FILE).rename(
    columns={
        "ccc": CLUSTERMATCH_LABEL,
        "pearson": PEARSON_LABEL,
        "spearman": SPEARMAN_LABEL,
    }
)

# %% tags=[]
df.shape

# %% tags=[]
df.head()

# %% [markdown] tags=[]
# ## Data stats

# %% tags=[]
df.describe().applymap(str)

# %% tags=[]
# skewness
df.apply(lambda x: stats.skew(x))

# %% [markdown] tags=[]
# # Histogram plot

# %% tags=[]
with sns.plotting_context("talk", font_scale=1.0):
    plot_histogram(df, output_dir=OUTPUT_FIGURE_DIR, fill=False)

# %% [markdown] tags=[]
# Coefficients' values distribute very differently. CCC is skewed to the left, whereas Pearson and specially Spearman seem more uniform.

# %% [markdown] tags=[]
# # Cumulative histogram plot

# %% [markdown] tags=[]
# I include also a cumulative histogram without specifying `bins`.

# %% tags=[]
with sns.plotting_context("talk", font_scale=1.0):
    plot_cumulative_histogram(df, GENE_PAIRS_PERCENT, output_dir=OUTPUT_FIGURE_DIR)

# %% [markdown] tags=[]
# # Joint plots comparing each coefficient

# %% tags=[]
with sns.plotting_context("talk", font_scale=1.0):
    jointplot(
        data=df,
        x=PEARSON_LABEL,
        y=CLUSTERMATCH_LABEL,
        add_corr_coefs=False,
        output_dir=OUTPUT_FIGURE_DIR,
    )

# %% tags=[]
with sns.plotting_context("talk", font_scale=1.0):
    x, y = SPEARMAN_LABEL, CLUSTERMATCH_LABEL

    g = jointplot(
        data=df,
        x=x,
        y=y,
        add_corr_coefs=False,
    )

    sns.despine(ax=g.ax_joint, left=True)
    g.ax_joint.set_yticks([])
    g.ax_joint.set_ylabel(None)

    g.savefig(
        OUTPUT_FIGURE_DIR / f"dist-{x.lower()}_vs_{y.lower()}.svg",
        bbox_inches="tight",
        dpi=300,
        facecolor="white",
    )

# %% tags=[]
with sns.plotting_context("talk", font_scale=1.0):
    jointplot(
        data=df,
        x=SPEARMAN_LABEL,
        y=PEARSON_LABEL,
        add_corr_coefs=False,
        output_dir=OUTPUT_FIGURE_DIR,
    )

# %% [markdown] tags=[]
# # Create final figure

# %% tags=[]
from svgutils.compose import Figure, SVG, Panel, Text

# %% tags=[]
Figure(
    "643.71cm",
    "427.66cm",
    Panel(
        SVG(OUTPUT_FIGURE_DIR / "dist-histograms.svg").scale(0.5),
        Text("a)", 2, 10, size=9, weight="bold"),
    ),
    Panel(
        SVG(OUTPUT_FIGURE_DIR / "dist-cum_histograms.svg").scale(0.5),
        Text("b)", 2, 10, size=9, weight="bold"),
    ).move(320, 0),
    Panel(
        SVG(OUTPUT_FIGURE_DIR / "dist-pearson_vs_ccc.svg").scale(0.595),
        Panel(
            SVG(OUTPUT_FIGURE_DIR / "dist-spearman_vs_ccc.svg")
            .scale(0.595)
            .move(215, 0)
        ),
        Panel(
            SVG(OUTPUT_FIGURE_DIR / "dist-spearman_vs_pearson.svg")
            .scale(0.595)
            .move(430, 0)
        ),
        Text("c)", 2, 10, size=9, weight="bold"),
    ).move(0, 220),
).save(OUTPUT_FIGURE_DIR / "dist-main.svg")

# %% [markdown] tags=[]
# Now open `dist-main.svg`, reside to fit drawing to page, and add a white rectangle to the background.

# %% tags=[]
