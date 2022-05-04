# This script installs R packages. When installing BiocManager, the script updates all R packages
# currently installed (options update=TRUE, ask=FALSE in BiocManager::install).


default_repo <- "http://cran.us.r-project.org"

# install BiocManager but do not update R packages so we keep those installed
# with conda
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = default_repo)
}
BiocManager::install(version = "3.13", update = FALSE, ask = FALSE)

# styler
BiocManager::install("styler", update = FALSE, ask = FALSE)

# org.Hs.eg.db
BiocManager::install("org.Hs.eg.db", update = FALSE, ask = FALSE)

# clusterProfiler
# BiocManager::install("clusterProfiler", update = FALSE, ask = FALSE)

# ReactomePA
# BiocManager::install("ReactomePA", update = FALSE, ask = FALSE)

# library(devtools)

# fgsea
# install_github("ctlab/fgsea", ref="v1.17.0")
