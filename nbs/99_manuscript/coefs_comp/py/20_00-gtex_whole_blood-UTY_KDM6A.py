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
# This notebooks analyzes more closely the pattern between gene pair *UTY* / *KDM6A*. The analyses are focused on the Reviewer 2's comment:
#
# ```
# In Figure 4, while there is a visible difference between the correlation of male samples, the CCC values are still quite close. For example, this can be observed in Brain Cerebellum and Small Intestine Terminal Ileum. Please address this.
# ```

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import pandas as pd

from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from ccc import conf
from ccc.coef import ccc

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
# this gene pair was originally found with ccc on whole blood
# interesting: https://clincancerres.aacrjournals.org/content/26/21/5567.figures-only
gene0_id, gene1_id = "ENSG00000147050.14", "ENSG00000183878.15"
gene0_symbol, gene1_symbol = "KDM6A", "UTY"

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
TISSUE_DIR = conf.GTEX["DATA_DIR"] / "data_by_tissue"
assert TISSUE_DIR.exists()

# %% tags=[]
OUTPUT_FIGURE_DIR = (
    conf.MANUSCRIPT["FIGURES_DIR"]
    / "coefs_comp"
    / f"{gene0_symbol.lower()}_vs_{gene1_symbol.lower()}"
)
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_FIGURE_DIR)

# %% [markdown] tags=[]
# # Data

# %% [markdown] tags=[]
# ## GTEx metadata

# %% tags=[]
gtex_metadata = pd.read_pickle(conf.GTEX["DATA_DIR"] / "gtex_v8-sample_metadata.pkl")

# %% tags=[]
gtex_metadata.shape

# %% tags=[]
gtex_metadata.head()

# %% [markdown] tags=[]
# ## Gene Ensembl ID -> Symbol mapping

# %% tags=[]
gene_map = pd.read_pickle(conf.GTEX["DATA_DIR"] / "gtex_gene_id_symbol_mappings.pkl")

# %% tags=[]
gene_map = gene_map.set_index("gene_ens_id")["gene_symbol"].to_dict()

# %% tags=[]
assert gene_map["ENSG00000145309.5"] == "CABS1"

# %% tags=[]
assert gene_map[gene0_id] == gene0_symbol
assert gene_map[gene1_id] == gene1_symbol

# %% [markdown] tags=[]
# ## Get male/females sample IDs

# %% tags=[]
gtex_metadata["SEX"].describe()

# %% tags=[]
male_samples = gtex_metadata[gtex_metadata["SEX"] == "Male"].index.tolist()

# %% tags=[]
len(male_samples)

# %% tags=[]
male_samples[:5]

# %% tags=[]
female_samples = gtex_metadata[gtex_metadata["SEX"] == "Female"].index.tolist()

# %% tags=[]
len(female_samples)

# %% tags=[]
female_samples[:5]

# %% [markdown] tags=[]
# # Brain cerebellum

# %% tags=[]
brain_cerebellum = (
    pd.read_pickle(TISSUE_DIR / "gtex_v8_data_brain_cerebellum.pkl")
    .loc[[gene0_id, gene1_id]]
    .T.rename_axis("sample_id")
)

# %% tags=[]
brain_cerebellum.shape

# %% tags=[]
brain_cerebellum.head()

# %% tags=[]
brain_cerebellum_males = brain_cerebellum.loc[
    brain_cerebellum.index.intersection(male_samples)
]

# %% tags=[]
brain_cerebellum_males.shape

# %% tags=[]
brain_cerebellum_females = brain_cerebellum.loc[
    brain_cerebellum.index.intersection(female_samples)
]

# %% tags=[]
brain_cerebellum_females.shape

# %% [markdown] tags=[]
# # Small intestine (terminal ileum)

# %% tags=[]
small_intestine = (
    pd.read_pickle(TISSUE_DIR / "gtex_v8_data_small_intestine_terminal_ileum.pkl")
    .loc[[gene0_id, gene1_id]]
    .T.rename_axis("sample_id")
)

# %% tags=[]
small_intestine.shape

# %% tags=[]
small_intestine.head()

# %% tags=[]
small_intestine_males = small_intestine.loc[
    small_intestine.index.intersection(male_samples)
]

# %% tags=[]
small_intestine_males.shape

# %% tags=[]
small_intestine_females = small_intestine.loc[
    small_intestine.index.intersection(female_samples)
]

# %% tags=[]
small_intestine_females.shape

# %% [markdown] tags=[]
# # Compute correlation

# %% [markdown] tags=[]
# ## Brain cerebellum

# %% [markdown] tags=[]
# ### CCC

# %% tags=[]
ccc(brain_cerebellum_males, pvalue_n_perms=1000)

# %% tags=[]
ccc(brain_cerebellum_females, pvalue_n_perms=1000)

# %% [markdown] tags=[]
# ### Pearson

# %% tags=[]
pearsonr(brain_cerebellum_males.iloc[:, 0], brain_cerebellum_males.iloc[:, 1])

# %% tags=[]
pearsonr(brain_cerebellum_females.iloc[:, 0], brain_cerebellum_females.iloc[:, 1])

# %% [markdown] tags=[]
# ### Spearman

# %% tags=[]
spearmanr(brain_cerebellum_males.iloc[:, 0], brain_cerebellum_males.iloc[:, 1])

# %% tags=[]
spearmanr(brain_cerebellum_females.iloc[:, 0], brain_cerebellum_females.iloc[:, 1])

# %% [markdown] tags=[]
# ## Small intestine (terminal ileum)

# %% [markdown] tags=[]
# ### CCC

# %% tags=[]
ccc(small_intestine_males, pvalue_n_perms=1000)

# %% tags=[]
ccc(small_intestine_females, pvalue_n_perms=1000)

# %% [markdown] tags=[]
# ### Pearson

# %% tags=[]
pearsonr(small_intestine_males.iloc[:, 0], small_intestine_males.iloc[:, 1])

# %% tags=[]
pearsonr(small_intestine_females.iloc[:, 0], small_intestine_females.iloc[:, 1])

# %% [markdown] tags=[]
# ### Spearman

# %% tags=[]
spearmanr(small_intestine_males.iloc[:, 0], small_intestine_males.iloc[:, 1])

# %% tags=[]
spearmanr(small_intestine_females.iloc[:, 0], small_intestine_females.iloc[:, 1])

# %% [markdown] tags=[]
# # Compute correlation on all tissues, males only

# %% tags=[]
res_all_males = pd.DataFrame(
    {
        f.stem.split("_data_")[1]: {
            "ccc": ccc(data[gene0_id], data[gene1_id]),
            "pearson": pearsonr(data[gene0_id], data[gene1_id])[0],
            "spearman": spearmanr(data[gene0_id], data[gene1_id])[0],
        }
        for f in TISSUE_DIR.glob("*.pkl")
        if (
            data := pd.read_pickle(f)
            .T[[gene0_id, gene1_id]]
            .reindex(male_samples)
            .dropna()
        )
        is not None
        and data.shape[0] > 10
    }
).T.abs()

# %% tags=[]
res_all_males.shape

# %% tags=[]
res_all_males.head()

# %% tags=[]
res_all_males.sort_values("ccc")

# %% tags=[]
res_all_males.sort_values("pearson")

# %% tags=[]
res_all_males.sort_values("spearman")


# %% [markdown] tags=[]
# # Plot of male samples

# %% tags=[]
def get_tissue_file(name):
    """
    Given a part of a tissue name, it returns a file path to the
    expression data for that tissue in GTEx. It fails if more than
    one files are found.

    Args:
        name: a string with the tissue name (or a part of it).

    Returns:
        A Path object pointing to the gene expression file for the
        given tissue.
    """
    tissue_files = []
    for f in TISSUE_DIR.glob("*.pkl"):
        if name in f.name:
            tissue_files.append(f)

    assert len(tissue_files) == 1
    return tissue_files[0]


# %% tags=[]
# testing
_tmp = get_tissue_file("whole_blood")
assert _tmp.exists()


# %% tags=[]
def simplify_tissue_name(tissue_name):
    return f"{tissue_name[0].upper()}{tissue_name[1:].replace('_', ' ')}"


# %% tags=[]
assert simplify_tissue_name("whole_blood") == "Whole blood"
assert simplify_tissue_name("uterus") == "Uterus"


# %% tags=[]
def plot_gene_pair(
    tissue_name,
    gene0,
    gene1,
    hue=None,
    kind="hex",
    ylim=None,
    bins="log",
    samples=None,
    filename_suffix="",
    tissue_name_in_title=None,
):
    """
    It plots (joint plot) a gene pair from the given tissue. It saves the plot
    for the manuscript.
    """
    # merge gene expression with metadata
    tissue_file = get_tissue_file(tissue_name)
    if samples is not None:
        tissue_data = (
            pd.read_pickle(tissue_file).T[[gene0, gene1]].reindex(samples).dropna()
        )
    else:
        tissue_data = pd.read_pickle(tissue_file).T[[gene0, gene1]]

    tissue_data = pd.merge(
        tissue_data,
        gtex_metadata,
        how="inner",
        left_index=True,
        right_index=True,
        validate="one_to_one",
    )

    # get gene symbols
    gene0_symbol, gene1_symbol = gene_map[gene0], gene_map[gene1]
    display((gene0_symbol, gene1_symbol))

    # compute correlations for this gene pair
    _clustermatch = ccc(tissue_data[gene0], tissue_data[gene1])
    _pearson = pearsonr(tissue_data[gene0], tissue_data[gene1])[0]
    _spearman = spearmanr(tissue_data[gene0], tissue_data[gene1])[0]

    if tissue_name_in_title is None:
        tissue_name_in_title = simplify_tissue_name(tissue_name)
    _title = f"{tissue_name_in_title}\n$c={_clustermatch:.2f}$  $p={_pearson:.2f}$  $s={_spearman:.2f}$"

    other_args = {
        "kind": kind,  # if hue is None else "scatter",
        "rasterized": True,
    }
    if hue is None:
        other_args["hue_order"] = None
    else:
        if tissue_data[hue].unique().shape[0] == 2:
            other_args["hue_order"] = ["Male", "Female"]

    with sns.plotting_context("paper", font_scale=1.5):
        p = sns.jointplot(
            data=tissue_data,
            x=gene0,
            y=gene1,
            hue=hue,
            **other_args,
            # ylim=(0, 500),
        )

        # if samples is not None:
        #     p.ax_joint.legend_.remove()

        if ylim is not None:
            p.ax_joint.set_ylim(ylim)

        gene_x_id = p.ax_joint.get_xlabel()
        gene_x_symbol = gene_map[gene_x_id]
        p.ax_joint.set_xlabel(f"{gene_x_symbol}", fontstyle="italic")

        gene_y_id = p.ax_joint.get_ylabel()
        gene_y_symbol = gene_map[gene_y_id]
        p.ax_joint.set_ylabel(f"{gene_y_symbol}", fontstyle="italic")

        p.fig.suptitle(_title)

        # save
        output_file = (
            OUTPUT_FIGURE_DIR
            / f"gtex_{tissue_name}-{gene_x_symbol}_vs_{gene_y_symbol}{filename_suffix}.svg"
        )
        display(output_file)

        plt.savefig(
            output_file,
            bbox_inches="tight",
            dpi=300,
            facecolor="white",
        )

    return tissue_data


# %% [markdown] tags=[]
# ## Brain cerebellum (males)

# %% tags=[]
_tissue_data = plot_gene_pair(
    "brain_cerebellum",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
    samples=None,
    filename_suffix="-all",
)

# %% tags=[]
_tissue_data = plot_gene_pair(
    "brain_cerebellum",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
    samples=male_samples,
    filename_suffix="-males",
)

# %% [markdown] tags=[]
# ## Smalle intestine (males)

# %% tags=[]
_tissue_data = plot_gene_pair(
    "small_intestine_terminal_ileum",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
    samples=None,
    filename_suffix="-all",
)

# %% tags=[]
_tissue_data = plot_gene_pair(
    "small_intestine_terminal_ileum",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
    samples=male_samples,
    filename_suffix="-males",
)

# %% [markdown] tags=[]
# # Understand how CCC divides samples

# %% [markdown] tags=[]
# ## Prepare datasets

# %% tags=[]
datasets_df = pd.DataFrame(
    {
        "dataset": "Brain cerebellum\n(males + females)",
        gene0_symbol: brain_cerebellum.iloc[:, 0],
        gene1_symbol: brain_cerebellum.iloc[:, 1],
    }
)

datasets_df = datasets_df.append(
    pd.DataFrame(
        {
            "dataset": "Small intestine (terminal ileum)\n(males + females)",
            gene0_symbol: small_intestine.iloc[:, 0],
            gene1_symbol: small_intestine.iloc[:, 1],
        }
    ),
    ignore_index=True,
)

datasets_df = datasets_df.append(
    pd.DataFrame(
        {
            "dataset": "Brain cerebellum\n(males)",
            gene0_symbol: brain_cerebellum_males.iloc[:, 0],
            gene1_symbol: brain_cerebellum_males.iloc[:, 1],
        }
    ),
    ignore_index=True,
)

datasets_df = datasets_df.append(
    pd.DataFrame(
        {
            "dataset": "Small intestine (terminal ileum)\n(males)",
            gene0_symbol: small_intestine_males.iloc[:, 0],
            gene1_symbol: small_intestine_males.iloc[:, 1],
        }
    ),
    ignore_index=True,
)

# %% tags=[]
datasets = {
    idx: df.drop(columns="dataset") for idx, df in datasets_df.groupby("dataset")
}


# %% tags=[]
def get_cm_line_points(x, y, max_parts, parts):
    """
    Given two data vectors (x and y) and the max_parts and parts
    returned from calling cm, this function returns two arrays with
    scalars to draw the lines that separates clusters in x and y.
    """
    # get the ccc partitions that maximize the coefficient
    x_max_part = parts[0][max_parts[0]]
    x_unique_k = {}
    for k in np.unique(x_max_part):
        data = x[x_max_part == k]
        x_unique_k[k] = data.min(), data.max()
    x_unique_k = sorted(x_unique_k.items(), key=lambda x: x[1][0])

    y_max_part = parts[1][max_parts[1]]
    y_unique_k = {}
    for k in np.unique(y_max_part):
        data = y[y_max_part == k]
        y_unique_k[k] = data.min(), data.max()
    y_unique_k = sorted(y_unique_k.items(), key=lambda x: x[1][0])

    x_line_points, y_line_points = [], []

    for idx in range(len(x_unique_k) - 1):
        k, (k_min, k_max) = x_unique_k[idx]
        nk, (nk_min, nk_max) = x_unique_k[idx + 1]

        x_line_points.append((k_max + nk_min) / 2.0)

    for idx in range(len(y_unique_k) - 1):
        k, (k_min, k_max) = y_unique_k[idx]
        nk, (nk_min, nk_max) = y_unique_k[idx + 1]

        y_line_points.append((k_max + nk_min) / 2.0)

    return x_line_points, y_line_points


# %% [markdown] tags=[]
# ## Brain cerebellum and Small intestine (males)

# %% tags=[]
with sns.plotting_context("paper", font_scale=1.8):
    g = sns.FacetGrid(
        data=datasets_df,
        col="dataset",
        col_order=[
            # "Brain cerebellum (all)",
            # "Small intestine (terminal ileum) (all)",
            "Brain cerebellum\n(males)",
            "Small intestine (terminal ileum)\n(males)",
        ],
        col_wrap=2,
        height=5,
    )
    g.map(sns.scatterplot, gene0_symbol, gene1_symbol, s=50, alpha=1)
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    for ds, ax in g.axes_dict.items():
        df = datasets[ds].to_numpy()
        x, y = df[:, 0], df[:, 1]

        # pearson and spearman
        r = pearsonr(x, y)[0]
        rs = spearmanr(x, y)[0]

        # ccc
        c, max_parts, parts = ccc(x, y, return_parts=True)
        c = ccc(x, y)

        x_line_points, y_line_points = get_cm_line_points(x, y, max_parts, parts)
        for yp in y_line_points:
            ax.hlines(y=yp, xmin=-0.5, xmax=30, color="r", alpha=0.5)

        for xp in x_line_points:
            ax.vlines(x=xp, ymin=-0.5, ymax=18, color="r", alpha=0.5)

        # add text box for the statistics
        stats = (
            f"$\it{{p}}$ ={r: .2f}\n"
            f"$\it{{s}}$ ={rs: .2f}\n"
            f"$\it{{c}}$ ={c: .2f}"
        )
        # stats = f"$c$ = {c:.2f}"
        bbox = dict(boxstyle="round", fc="white", ec="black", alpha=0.75)
        ax.text(
            0.95,
            0.75,
            stats,
            fontsize=14,
            bbox=bbox,
            transform=ax.transAxes,
            horizontalalignment="right",
        )
        ax.set_xlabel(f"{gene0_symbol}", fontstyle="italic")
        ax.set_ylabel(f"{gene1_symbol}", fontstyle="italic")
        ax.set(xticklabels=[], yticklabels=[])
        ax.tick_params(left=False, bottom=False)

    plt.savefig(
        OUTPUT_FIGURE_DIR
        / "gtex_brain_cerebellum_and_small_intestine_terminal_ileum-KDM6A_vs_UTY-males.svg",
        # rasterized=True,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

# %% [markdown] tags=[]
# ## Plot

# %% tags=[]
with sns.plotting_context("paper", font_scale=1.8):
    g = sns.FacetGrid(
        data=datasets_df,
        col="dataset",
        col_order=[
            # "Brain cerebellum (all)",
            # "Small intestine (terminal ileum) (all)",
            "Brain cerebellum\n(males)",
            "Small intestine (terminal ileum)\n(males)",
        ],
        col_wrap=2,
        height=5,
    )
    g.map(sns.scatterplot, gene0_symbol, gene1_symbol, s=50, alpha=1)
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    for ds, ax in g.axes_dict.items():
        df = datasets[ds].to_numpy()
        x, y = df[:, 0], df[:, 1]

        # pearson and spearman
        r = pearsonr(x, y)[0]
        rs = spearmanr(x, y)[0]

        # ccc
        c, max_parts, parts = ccc(x, y, return_parts=True)
        c = ccc(x, y)

        x_line_points, y_line_points = get_cm_line_points(x, y, max_parts, parts)
        for yp in y_line_points:
            ax.hlines(y=yp, xmin=-0.5, xmax=30, color="r", alpha=0.5)

        for xp in x_line_points:
            ax.vlines(x=xp, ymin=-0.5, ymax=18, color="r", alpha=0.5)

        # add text box for the statistics
        stats = f"$c$ = {c:.2f}"
        bbox = dict(boxstyle="round", fc="white", ec="black", alpha=0.75)
        ax.text(
            0.95,
            0.90,
            stats,
            fontsize=14,
            bbox=bbox,
            transform=ax.transAxes,
            horizontalalignment="right",
        )
        ax.set_xlabel(f"{gene0_symbol}", fontstyle="italic")
        ax.set_ylabel(f"{gene1_symbol}", fontstyle="italic")
        ax.set(xticklabels=[], yticklabels=[])
        ax.tick_params(left=False, bottom=False)

    plt.savefig(
        OUTPUT_FIGURE_DIR
        / "gtex-KDM6A_vs_UTY-brain_cerebellum_and_small_intestine_terminal_ileum-clusters-males.svg",
        # rasterized=True,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

# %% tags=[]
with sns.plotting_context("paper", font_scale=1.8):
    g = sns.FacetGrid(
        data=datasets_df,
        col="dataset",
        col_order=[
            "Brain cerebellum\n(males + females)",
            "Small intestine (terminal ileum)\n(males + females)",
            # "Brain cerebellum (males)",
            # "Small intestine (terminal ileum) (males)",
        ],
        col_wrap=2,
        height=5,
    )
    g.map(sns.scatterplot, gene0_symbol, gene1_symbol, s=50, alpha=1)
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    for ds, ax in g.axes_dict.items():
        df = datasets[ds].to_numpy()
        x, y = df[:, 0], df[:, 1]

        # pearson and spearman
        r = pearsonr(x, y)[0]
        rs = spearmanr(x, y)[0]

        # ccc
        c, max_parts, parts = ccc(x, y, return_parts=True)
        c = ccc(x, y)

        x_line_points, y_line_points = get_cm_line_points(x, y, max_parts, parts)
        for yp in y_line_points:
            ax.hlines(y=yp, xmin=-0.5, xmax=30, color="r", alpha=0.5)

        for xp in x_line_points:
            ax.vlines(x=xp, ymin=-0.5, ymax=18, color="r", alpha=0.5)

        # add text box for the statistics
        stats = f"$c$ = {c:.2f}"
        bbox = dict(boxstyle="round", fc="white", ec="black", alpha=0.75)
        ax.text(
            0.95,
            0.90,
            stats,
            fontsize=14,
            bbox=bbox,
            transform=ax.transAxes,
            horizontalalignment="right",
        )
        ax.set_xlabel(f"{gene0_symbol}", fontstyle="italic")
        ax.set_ylabel(f"{gene1_symbol}", fontstyle="italic")
        ax.set(xticklabels=[], yticklabels=[])
        ax.tick_params(left=False, bottom=False)

    plt.savefig(
        OUTPUT_FIGURE_DIR
        / "gtex-KDM6A_vs_UTY-brain_cerebellum_and_small_intestine_terminal_ileum-clusters-all.svg",
        # rasterized=True,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

# %% [markdown] tags=[]
# # Create final figure

# %% tags=[]
from svgutils.compose import Figure, SVG, Panel, Text

# %% tags=[]
Figure(
    f"{60.0767480 * 6}cm",
    f"{60.0767480 * 6}cm",
    # Panel(
    #     SVG(OUTPUT_FIGURE_DIR / "gtex_brain_cerebellum_and_small_intestine_terminal_ileum-KDM6A_vs_UTY-males.svg").scale(0.5),
    #     Text("a)", 2, 10, size=9, weight="bold"),
    # ),
    Panel(
        SVG(OUTPUT_FIGURE_DIR / "gtex_brain_cerebellum_and_small_intestine_terminal_ileum-KDM6A_vs_UTY-males.svg").scale(0.5),
        Text("a)", 2, 10, size=9, weight="bold"),
    ),#.move(0, 180),
    Panel(
        SVG(OUTPUT_FIGURE_DIR / "gtex-KDM6A_vs_UTY-brain_cerebellum_and_small_intestine_terminal_ileum-clusters-all.svg").scale(0.5),
        Text("b)", 2, 10, size=9, weight="bold"),
    ).move(0, 170),#.move(0, 180+170),
).save(OUTPUT_FIGURE_DIR / "gtex-KDM6A_vs_UTY-nonlinear_and_linear.svg")

# %% [markdown] tags=[]
# Now open the final file, reside to fit drawing to page, and add a white rectangle to the background.

# %% tags=[]
