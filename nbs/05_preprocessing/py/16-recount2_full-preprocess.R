# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     text_representation:
#       extension: .R
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# This notebook contains code taken from the MultiPLIER repo to download and process recount2 data.
# The code was taken from [here](https://github.com/greenelab/rheum-plier-data/blob/master/recount2/1-get_all_recount_dataset.R) and [here](https://github.com/greenelab/rheum-plier-data/blob/master/recount2/2-prep_recount_for_plier.R).
#
# The output are Python pickle files with a large matrix with genes in rows and samples in columns, and another file with gene ID mappings.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
`%>%` <- dplyr::`%>%`
library(biomaRt)
library(reticulate)

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
recount2full.data.dir <- Sys.getenv("CM_RECOUNT2FULL_DATA_DIR")

# %% tags=[]
recount2full.data.dir

# %% tags=[]
dir.create(recount2full.data.dir, recursive = TRUE, showWarnings = FALSE)

# %% tags=[]
data.dir <- Sys.getenv("CM_RECOUNT2FULL_INTERNAL_DATA_DIR")

# %% tags=[]
data.dir

# %% tags=[]
dir.create(data.dir, recursive = TRUE, showWarnings = FALSE)

# %% [markdown] tags=[]
# # Load raw data

# %% tags=[]
input.file <- file.path(recount2full.data.dir, "recount2_rpkm_raw.rds")

# %% tags=[]
input.file

# %% tags=[]
rpkm.df <- readRDS(input.file)

# %% tags=[]
dim(rpkm.df)

# %% tags=[]
head(rpkm.df[, 1:10])

# %% [markdown] tags=[]
# # Preprocess data

# %% tags=[]
# Transform ensembl id to genesymbol
mart <- biomaRt::useDataset(
  "hsapiens_gene_ensembl",
  biomaRt::useMart("ensembl")
)

# %% tags=[]
genes <- unlist(lapply(strsplit(rpkm.df$ENSG, "[.]"), `[[`, 1))

rpkm.df$ensembl_gene_id <- unlist(lapply(
  strsplit(rpkm.df$ENSG, "[.]"),
  `[[`, 1
))

gene.df <- biomaRt::getBM(
  filters = "ensembl_gene_id",
  attributes = c("ensembl_gene_id", "hgnc_symbol"),
  values = genes,
  mart = mart
)

# %% tags=[]
# filter to remove genes without a gene symbol
gene.df <- gene.df %>% dplyr::filter(complete.cases(.))

# %% tags=[]
# add gene symbols to expression df
rpkm.df <- dplyr::inner_join(gene.df, rpkm.df,
  by = "ensembl_gene_id"
)

# %% tags=[]
# keep gene mappings
gene.df <- rpkm.df %>% dplyr::select(ensembl_gene_id, hgnc_symbol)

# %% tags=[]
dim(gene.df)

# %% tags=[]
head(gene.df)

# %% tags=[]
# set Ensemble IDs as rownames
rownames(rpkm.df) <- make.names(rpkm.df$ensembl_gene_id, unique = TRUE)

# %% tags=[]
# remove gene identifier columns
rpkm.df <- rpkm.df %>% dplyr::select(-c(ensembl_gene_id:ENSG))

# %% tags=[]
dim(rpkm.df)

# %% tags=[]
head(rpkm.df[, 1:10])

# %% [markdown] tags=[]
# # Save

# %% [markdown] tags=[]
# ## Gene ID mappings

# %% tags=[]
output_filepath <- file.path(recount2full.data.dir, "recount2_gene_ids_mappings")

# %% tags=[]
output_filepath

# %% tags=[]
saveRDS(gene.df, file = paste0(output_filepath, ".rds"))

# %% tags=[]
py_save_object(gene.df, paste0(output_filepath, ".pkl"))

# %% [markdown] tags=[]
# ## Gene expression data

# %% tags=[]
output_filepath <- file.path(recount2full.data.dir, "recount2_rpkm")

# %% tags=[]
output_filepath

# %% tags=[]
saveRDS(rpkm.df, file = paste0(output_filepath, ".rds"))

# %% tags=[]
py_save_object(rpkm.df, paste0(output_filepath, ".pkl"))

# %% [markdown] tags=[]
# # Cleanup

# %% tags=[]
# the raw file is not longer necessary
if (file.exists(input.file)) {
  # Delete file if it exists
  file.remove(input.file)
}
