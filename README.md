# Clustermatch on gene expression data (code)

[![Code tests](https://github.com/greenelab/clustermatch-gene-expr/actions/workflows/pytest.yaml/badge.svg)](https://github.com/greenelab/clustermatch-gene-expr/actions/workflows/pytest.yaml)
[![codecov](https://codecov.io/gh/greenelab/clustermatch-gene-expr/branch/main/graph/badge.svg?token=QNK6O3Y1VF)](https://codecov.io/gh/greenelab/clustermatch-gene-expr)
[![HTML Manuscript](https://img.shields.io/badge/manuscript-HTML-blue.svg)](https://greenelab.github.io/clustermatch-gene-expr-manuscript/)
[![PDF Manuscript](https://img.shields.io/badge/manuscript-PDF-blue.svg)](https://greenelab.github.io/clustermatch-gene-expr-manuscript/manuscript.pdf)


## Overview

![](images/cm_gene_expr_overview.png)

TODO: update description and links to manuscripts

This repository contains the source code to reproduce the analyses of Clustermatch on gene expression data.
If you want to use Clustermatch as a standalone tool to perform your own analyses, please go to the [official repository](https://github.com/sinc-lab/clustermatch) and follow the installation instructions.

For more details, check out our manuscript in COMPLETE or our [Manubot web version](https://greenelab.github.io/clustermatch-gene-expr-manuscript/).


## Setup

To prepare the environment to run the analyses, follow the steps in [environment](environment/).
This will create a conda environment and download the necessary data.
Alternatively, you can use our Docker image (see below).

## Running code

### From command-line

First, activate your conda environment and export your settings to environmental variables so non-Python scripts can access them:
```bash
conda activate clustermatch_gene_expr
eval `python libs/conf.py`
```

The code to preprocess data and generate results is in the `nbs/` folder.
All notebooks are organized by directories, such as `01_preprocessing`, with file names that indicate the order in which they should be run (if they share the prefix, then it means they can be run in parallel).
For example, to run all notebooks for the preprocessing step, you can use this command (requires [GNU Parallel](https://www.gnu.org/software/parallel/)):

```bash
cd nbs/
parallel -k --lb --halt 2 -j1 'bash run_nbs.sh {}' ::: 01_preprocessing/*.ipynb
```

<!--
Or if you want to run all the analyses at once, you can use:

```bash
shopt -s globstar
parallel -k --lb --halt 2 -j1 'bash run_nbs.sh {}' ::: nbs/{,**/}*.ipynb
```
-->

### From your browser

Alternatively, you can start your JupyterLab server by running:

```bash
bash scripts/run_nbs_server.sh
```

Then, go to `http://localhost:8892`, browse the `nbs` folder, and run the notebooks in the specified order.

## Using Docker

You can also run all the steps below using a Docker image instead of a local installation.

```bash
docker pull miltondp/clustermatch_gene_expr
```

The image only contains the conda environment with the code in this repository so, after pulling the image, you need to download the data as well:

```bash
mkdir -p /tmp/clustermatch_gene_expr_data

docker run --rm \
  -v "/tmp/clustermatch_gene_expr_data:/opt/data" \
  --user "$(id -u):$(id -g)" \
  miltondp/clustermatch_gene_expr \
  /bin/bash -c "python environment/scripts/setup_data.py"
```

The `-v` parameter allows specifying a local directory (`/tmp/clustermatch_gene_expr_data`) where the data will be downloaded.
If you want to generate the figures and tables for the manuscript, you need to clone the [manuscript repo](https://github.com/greenelab/clustermatch-gene-expr-manuscript) and pass it with `-v [PATH_TO_MANUSCRIPT_REPO]:/opt/manuscript`.
If you want to change any other setting, you can set environmental variables when running the container; for example, to change the number of cores used to 2: `-e CM_N_JOBS=2`.

You can run notebooks from the command line, for example:

```bash
docker run --rm \
  -v "/tmp/clustermatch_gene_expr_data:/opt/data" \
  --user "$(id -u):$(id -g)" \
  miltondp/clustermatch_gene_expr \
  /bin/bash -c "parallel -k --lb --halt 2 -j1 'bash nbs/run_nbs.sh {}' ::: nbs/05_preprocessing/*.ipynb"
```

or start a Jupyter Notebook server with:

```bash
docker run --rm \
  -p 8888:8893 \
  -v "/tmp/clustermatch_gene_expr_data:/opt/data" \
  --user "$(id -u):$(id -g)" \
  miltondp/clustermatch_gene_expr
```

and access the interface by going to `http://localhost:8888`.
