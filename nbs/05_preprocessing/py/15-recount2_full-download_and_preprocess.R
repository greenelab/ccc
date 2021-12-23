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
library(recount)
library(biomaRt)
library(reticulate)

# %% [markdown] tags=[]
# # Settings

# %%
recount2full.data.dir <- Sys.getenv("CM_RECOUNT2FULL_DATA_DIR")

# %%
recount2full.data.dir

# %%
dir.create(recount2full.data.dir, recursive = TRUE, showWarnings = FALSE)

# %%
data.dir <- Sys.getenv("CM_RECOUNT2FULL_INTERNAL_DATA_DIR")

# %%
data.dir

# %% tags=[]
dir.create(data.dir, recursive = TRUE, showWarnings = FALSE)

# %% [markdown] tags=[]
# # Functions

# %%
# Get RPKM value for each gene - adapted from recount package
getRPKM <- function(rse, length_var = "bp_length", mapped_var = NULL) {
  # Computes the RPKM value for each gene in the sample.
  #
  # Args:
  #  rse: A RangedSummarizedExperiment-class object in recount package
  #  length_var: A length 1 character vector with the column name from rowData(rse) that has
  #              the coding length. For gene level objects from recount this is bp_length. If
  #              NULL, then it will use width(rowRanges(rse)) which should be used for exon RSEs.
  #  mapped_var: A length 1 character vector with the column name from colData(rse) that has
  #              the number of reads mapped. If NULL (default) then it will use the column
  #              sums of the counts matrix
  # Returns:
  #   RPKM value for each sample
  if (!is.null(mapped_var)) {
    mapped <- colData(rse)[, mapped_var]
  } else {
    mapped <- colSums(assays(rse)$counts)
  }
  bg <- matrix(mapped, ncol = ncol(rse), nrow = nrow(rse), byrow = TRUE)
  if (!is.null(length_var)) {
    len <- rowData(rse)[, length_var]
  } else {
    len <- width(rowRanges(rse))
  }
  wid <- matrix(len, nrow = nrow(rse), ncol = ncol(rse), byrow = FALSE)
  rpkm <- assays(rse)$counts / (wid / 1000) / (bg / 1e6)
  return(rpkm)
}

# %% [markdown] tags=[]
# # Download

# %%
# Get all samples from recount database
metasample.sra <- all_metadata(subset = "sra", verbose = TRUE)
metasample.sra <- as.data.frame(metasample.sra)

# %%
# Remove samples without description
metadata.nonempty <- metasample.sra[!is.na(metasample.sra$characteristics), ]
included.sample.list <- unique(metadata.nonempty$project)

# %%
# Download all recount2 samples in included.sample.list
lapply(
  included.sample.list,
  function(x) {
    download_study(x,
      type = "rse-gene",
      outdir = file.path(data.dir, x)
    )
  }
)

# %% [markdown] tags=[]
# # Normalize with RPKM

# %%
# get RPKM for each experiment and add to list
rpkm.list <- list()
for (experiment in included.sample.list) {
  load(file.path(data.dir, experiment, "rse_gene.Rdata"))
  rpkm <- as.data.frame(getRPKM(rse_gene))
  rpkm$id <- rownames(rpkm)
  rpkm.list[[experiment]] <- rpkm
}

# %%
# combine experiments -- this is the most memory efficient way to go about this
# that I've found -- will need to drop extraneous gene id columns
rpkm.df <- do.call(base::cbind, c(rpkm.list, by = "id"))
rpkm.df <- rpkm.df %>% dplyr::select(-dplyr::ends_with("id"))
rpkm.df <- tibble::rownames_to_column(rpkm.df, "ENSG")
# drop last column "by" -- information about what was used with base::cbind
rpkm.df <- rpkm.df %>% dplyr::select(-by)

# %% [markdown] tags=[]
# # Preprocess data

# %%
# Transform ensembl id to genesymbol
mart <- biomaRt::useDataset(
  "hsapiens_gene_ensembl",
  biomaRt::useMart("ensembl")
)
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

# %%
# filter to remove genes without a gene symbol
gene.df <- gene.df %>% dplyr::filter(complete.cases(.))

# %%
# add gene symbols to expression df
rpkm.df <- dplyr::inner_join(gene.df, rpkm.df,
  by = "ensembl_gene_id"
)

# %%
# keep gene mappings
gene.df <- rpkm.df %>% dplyr::select(ensembl_gene_id, hgnc_symbol)

# %%
dim(gene.df)

# %%
head(gene.df)

# %%
# set Ensemble IDs as rownames
rownames(rpkm.df) <- make.names(rpkm.df$ensembl_gene_id, unique = TRUE)

# %%
# remove gene identifier columns
rpkm.df <- rpkm.df %>% dplyr::select(-c(ensembl_gene_id:ENSG))

# %%
dim(rpkm.df)

# %%
head(rpkm.df[, 1:10])

# %% [markdown] tags=[]
# # Save

# %% [markdown] tags=[]
# ## Gene ID mappings

# %%
output_filepath <- file.path(recount2full.data.dir, "recount2_gene_ids_mappings")

# %%
output_filepath

# %%
saveRDS(gene.df, file = paste0(output_filepath, ".rds"))

# %%
py_save_object(gene.df, paste0(output_filepath, ".pkl"))

# %% [markdown] tags=[]
# ## Gene expression data

# %%
output_filepath <- file.path(recount2full.data.dir, "recount2_rpkm")

# %%
output_filepath

# %%
saveRDS(rpkm.df, file = paste0(output_filepath, ".rds"))

# %%
py_save_object(rpkm.df, paste0(output_filepath, ".pkl"))

# %%
