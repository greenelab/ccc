# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     formats: ipynb,py//py:percent
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
# This notebook reads 1) the normalized gene expression and 2) pathways from the data processed by
# MultiPLIER scripts (https://github.com/greenelab/multi-plier) and saves it into a more friendly Python
# format (Pandas DataFrames as pickle files).
#
# For recount2 we will not perform gene selection as in GTEx, since in this dataset (from MUltiPLIER) we only have ~6,700 genes, which is managable enough to compute the similarity matrices.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import sys
from pathlib import Path
from shutil import copyfile

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from clustermatch import conf

# %% tags=[]
readRDS = ro.r["readRDS"]

# %% tags=[]
saveRDS = ro.r["saveRDS"]

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
OUTPUT_DIR = conf.RECOUNT2["DATA_RDS_FILE"].parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %% [markdown] tags=[]
# # Read entire recount data prep file with R

# %% tags=[]
display(conf.RECOUNT2["DATA_RDS_FILE"])

if not conf.RECOUNT2["DATA_RDS_FILE"].exists():
    print("Input file does not exist")
    sys.exit(0)

# %% tags=[]
recount_data_prep = readRDS(str(conf.RECOUNT2["DATA_RDS_FILE"]))

# %% [markdown] tags=[]
# # recount2 gene expression data

# %% [markdown] tags=[]
# ## Load

# %% tags=[]
recount2_rpkl_cm = recount_data_prep.rx2("rpkm.cm")

# %% tags=[]
recount2_rpkl_cm

# %% tags=[]
recount2_rpkl_cm.rownames

# %% tags=[]
recount2_rpkl_cm.colnames

# %% tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
    recount2_rpkl_cm = ro.conversion.rpy2py(recount2_rpkl_cm)

# %% tags=[]
assert recount2_rpkl_cm.shape == (6750, 37032)

# %% tags=[]
recount2_rpkl_cm.shape

# %% tags=[]
recount2_rpkl_cm.head()

# %% [markdown] tags=[]
# ## Test

# %% tags=[]
assert not recount2_rpkl_cm.isna().any().any()
assert recount2_rpkl_cm.index.is_unique
assert recount2_rpkl_cm.columns.is_unique

# %% [markdown] tags=[]
# Test whether what I load from a plain R session is the same as in here.

# %% tags=[]
assert recount2_rpkl_cm.loc["GAS6", "SRP000599.SRR013549"].round(4) == -0.3125

# %% tags=[]
assert recount2_rpkl_cm.loc["GAS6", "SRP045352.SRR1539229"].round(7) == -0.2843801

# %% tags=[]
assert recount2_rpkl_cm.loc["CFL2", "SRP056840.SRR1951636"].round(7) == -0.3412832

# %% tags=[]
assert recount2_rpkl_cm.iloc[9, 16].round(7) == -0.4938852

# %% [markdown] tags=[]
# ## Save

# %% [markdown] tags=[]
# ### Pickle format (binary)

# %% tags=[]
output_filename = conf.RECOUNT2["DATA_FILE"]

display(output_filename)

# %% tags=[]
recount2_rpkl_cm.to_pickle(output_filename)

# %% tags=[]
# delete the object to save memory
del recount2_rpkl_cm

# %% [markdown] tags=[]
# # recount2 pathways

# %% [markdown] tags=[]
# ## Load

# %% tags=[]
recount2_all_paths_cm = recount_data_prep.rx2("all.paths.cm")

# %% tags=[]
recount2_all_paths_cm

# %% tags=[]
recount2_all_paths_cm.rownames

# %% tags=[]
recount2_all_paths_cm.colnames

# %% tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
    recount2_all_paths_cm_values = ro.conversion.rpy2py(recount2_all_paths_cm)

# %% tags=[]
recount2_all_paths_cm_values

# %% tags=[]
recount2_all_paths_cm_df = pd.DataFrame(
    data=recount2_all_paths_cm_values,
    index=recount2_all_paths_cm.rownames,
    columns=recount2_all_paths_cm.colnames,
    dtype=bool,
)

# %% tags=[]
assert recount2_all_paths_cm_df.shape == (6750, 628)

# %% tags=[]
recount2_all_paths_cm_df.shape

# %% tags=[]
_tmp = recount2_all_paths_cm_df.dtypes.unique()
display(_tmp)
assert len(_tmp) == 1

# %% tags=[]
recount2_all_paths_cm_df.head()

# %% [markdown] tags=[]
# ## Test

# %% tags=[]
assert not recount2_all_paths_cm_df.loc[
    "CTSD", "REACTOME_SCFSKP2_MEDIATED_DEGRADATION_OF_P27_P21"
]

# %% tags=[]
assert recount2_all_paths_cm_df.loc["CTSD", "PID_P53DOWNSTREAMPATHWAY"]

# %% tags=[]
assert recount2_all_paths_cm_df.loc["MMP14", "PID_HIF2PATHWAY"]

# %% [markdown] tags=[]
# ## Save

# %% [markdown] tags=[]
# ### Pickle format

# %% tags=[]
output_filename = conf.RECOUNT2["DATA_FILE"].parent / "recount_all_paths_cm.pkl"
display(output_filename)

# %% tags=[]
recount2_all_paths_cm_df.to_pickle(output_filename)

# %% [markdown] tags=[]
# ### RDS format

# %% tags=[]
output_rds_file = output_filename.with_suffix(".rds")
display(output_rds_file)

# %% tags=[]
saveRDS(recount2_all_paths_cm, str(output_rds_file))

# %% [markdown] tags=[]
# ### Text format

# %% tags=[]
# tsv format
output_text_file = output_filename.with_suffix(".tsv.gz")
display(output_text_file)

# %% tags=[]
recount2_all_paths_cm_df.astype("int").head()

# %% tags=[]
recount2_all_paths_cm_df.astype("int").to_csv(output_text_file, sep="\t", index=True)

# %% tags=[]
