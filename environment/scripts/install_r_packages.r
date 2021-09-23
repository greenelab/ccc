# This script installs R packages. When installing BiocManager, the script updates all R packages
# currently installed (options update=TRUE, ask=FALSE in BiocManager::install).


default_repo = 'http://cran.us.r-project.org'

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager", repos=default_repo)
BiocManager::install(version = "3.13", update=TRUE, ask=FALSE)

# clusterProfiler
BiocManager::install("clusterProfiler", update=FALSE, ask=FALSE)

# library(devtools)

# fgsea
# install_github("ctlab/fgsea", ref="v1.17.0")
