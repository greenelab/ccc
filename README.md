# Clustermatch Correlation Coefficient (CCC)

[![Code tests](https://github.com/greenelab/ccc/actions/workflows/pytest.yaml/badge.svg)](https://github.com/greenelab/ccc/actions/workflows/pytest.yaml)
[![codecov](https://codecov.io/gh/greenelab/ccc/branch/main/graph/badge.svg?token=QNK6O3Y1VF)](https://codecov.io/gh/greenelab/ccc)
[![bioRxiv Manuscript](https://img.shields.io/badge/manuscript-bioRxiv-blue.svg)](https://doi.org/10.1101/2022.06.15.496326)
[![HTML Manuscript](https://img.shields.io/badge/manuscript-HTML-blue.svg)](https://greenelab.github.io/ccc-manuscript/)

## Overview

The Clustermatch Correlation Coefficient (CCC) is a highly-efficient, next-generation not-only-linear correlation coefficient that can work on numerical and categorical data types.
This repository contains the code of CCC and instructions to install and use it.
It also has all the scripts/notebooks to run the analyses for the [manuscript](https://github.com/greenelab/ccc-manuscript), where we applied CCC on gene expression data.

## Installation

CCC is available as a PyPI (Python) package (`ccc-coef`). We tested CCC in Python 3.9+, but it should work on prior 3.x versions.
You can quickly test it by creating a conda environment and then install it with `pip`:

```bash
# ipython and pandas are used in the following examples, but they are not needed for CCC to work
conda create -y -n ccc-env python=3.9 ipython pandas
conda activate ccc-env
pip install ccc-coef
```

## Usage

Run `ipython` in your terminal:
```bash
$ ipython
Python 3.10.4 (main, Mar 31 2022, 08:41:55) [GCC 7.5.0]
Type 'copyright', 'credits' or 'license' for more information
IPython 8.3.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: 
```

CCC can work on numerical data represented as a `numpy.array` or `pandas.Series`:

```python
In [1]: import numpy as np
In [2]: import pandas as pd
In [3]: from ccc.coef import ccc

In [4]: random_feature1 = np.random.rand(1000)
In [5]: random_feature2 = np.random.rand(1000)
In [6]: ccc(random_feature1, random_feature2)
Out[6]: 0.0018815884476534295

In [7]: random_feature1 = pd.Series(random_feature1)
In [8]: random_feature2 = pd.Series(random_feature2)
In [9]: ccc(random_feature1, random_feature2)
Out[9]: 0.0018815884476534295
```

CCC always returns a value between zero (no relationship) and one (perfect relationship).

You can also mix numerical and categorical data:

```python
In [10]: categories = np.array(["blue", "red", "green", "yellow"])
In [11]: categorical_random_feature1 = np.random.choice(categories, size=1000)
In [12]: categorical_random_feature2 = np.random.choice(categories, size=1000)
In [14]: categorical_random_feature2[:10]
Out[14]: 
array(['yellow', 'red', 'red', 'yellow', 'blue', 'blue', 'red', 'yellow',
       'green', 'blue'], dtype='<U6')

In [15]: ccc(categorical_random_feature1, categorical_random_feature2)
Out[15]: 0.0009263483455638076

In [17]: ccc(random_feature1, categorical_random_feature2)
Out[17]: 0.0015123522641692117
```

The first argument of `ccc` could also be a matrix, either as a `numpy.array` (features are in rows and objects in columns) or as a `pandas.DataFrame` (objects are in rows and features in columns).
In this case, `ccc` will compute the pairwise correlation across all features:

```python
In [18]: data = np.random.rand(10, 1000)

In [21]: # with a numpy.array
In [22]: c = ccc(data)
In [23]: c.shape
Out[23]: (45,)
In [24]: c[:10]
Out[24]: 
array([0.00404461, 0.00185342, 0.00248847, 0.00232761, 0.00260786,
       0.00121495, 0.00227679, 0.00099051, 0.00313611, 0.00323936])

In [25]: # with a pandas.DataFrame
In [26]: data_df = pd.DataFrame(data.T)

In [29]: c = ccc(data_df)
In [30]: c.shape
Out[30]: (45,)
In [31]: c[:10]
Out[31]: 
array([0.00404461, 0.00185342, 0.00248847, 0.00232761, 0.00260786,
       0.00121495, 0.00227679, 0.00099051, 0.00313611, 0.00323936])
```

If your data has a mix of numerical and categorical features, it's better to work on a `pandas.DataFrame`.
As an example, we load the [Titanic dataset](https://www.kaggle.com/c/titanic/data) (from [seaborn](https://github.com/mwaskom/seaborn-data/)'s repository):

```python
In [35]: titanic_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/raw/titanic.csv"
In [36]: titanic_df = pd.read_csv(titanic_url)
In [33]: titanic_df.shape
Out[33]: (891, 11)
In [34]: titanic_df.head()
Out[34]: 
   survived  pclass                                               name     sex   age  sibsp  parch            ticket     fare cabin embarked
0         0       3                            Braund, Mr. Owen Harris    male  22.0      1      0         A/5 21171   7.2500   NaN        S
1         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0          PC 17599  71.2833   C85        C
2         1       3                             Heikkinen, Miss. Laina  female  26.0      0      0  STON/O2. 3101282   7.9250   NaN        S
3         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0            113803  53.1000  C123        S
4         0       3                           Allen, Mr. William Henry    male  35.0      0      0            373450   8.0500   NaN        S
```

The Titanic dataset has missing values:

```python
Out[35]: titanic_df.isna().sum()
Out[35]: 
survived      0
pclass        0
name          0
sex           0
age         177
sibsp         0
parch         0
ticket        0
fare          0
cabin       687
embarked      2
dtype: int64
```

So we need some kind of preprocessing before moving on:

```python
In [62]: titanic_df = titanic_df.dropna(subset=["embarked"]).dropna(axis=1)
In [63]: titanic_df.shape
Out[63]: (889, 9)
```

Now we can run CCC on the dataset and get a correlation matrix across features:

```python
In [18]: ccc_corrs = ccc(titanic_df)

In [27]: from scipy.spatial.distance import squareform
In [22]: ccc_corrs = squareform(ccc_corrs)
In [23]: np.fill_diagonal(ccc_corrs, 1.0)
In [29]: ccc_corrs = pd.DataFrame(ccc_corrs, index=titanic_df.columns.tolist(), columns=titanic_df.columns.tolist())
In [25]: ccc_corrs.shape
Out[25]: (11, 11)
In [264]: with pd.option_context('display.float_format', '{:,.2f}'.format):
     ...:     display(ccc_corrs)
          survived  pclass  name  sex  sibsp  parch  ticket  fare  embarked
survived      1.00    0.12  0.00 0.32   0.04   0.05    0.00  0.07      0.05
pclass        0.12    1.00  0.00 0.04   0.02   0.01    0.00  0.33      0.01
name          0.00    0.00  1.00 0.00   0.00   0.00    0.00  0.00      0.00
sex           0.32    0.04  0.00 1.00   0.08   0.11    0.00  0.04      0.04
sibsp         0.04    0.02  0.00 0.08   1.00   0.29    0.00  0.23      0.00
parch         0.05    0.01  0.00 0.11   0.29   1.00    0.00  0.14      0.00
ticket        0.00    0.00  0.00 0.00   0.00   0.00    1.00  0.02      0.00
fare          0.07    0.33  0.00 0.04   0.23   0.14    0.02  1.00      0.03
embarked      0.05    0.01  0.00 0.04   0.00   0.00    0.00  0.03      1.00
```

As we show in the manuscript, the distribution of CCC values is much more skewed than other coefficients like Pearson's or Spearman's.
A comparison between these coefficients should account for that.

The `ccc` function also has a `n_jobs` parameter that allows to control the number of CPU cores used.
Below we compute the pairwise correlation between 20 features across 1000 objects:

```python
In [268]: data = np.random.rand(20, 1000)

In [269]: %timeit ccc(data, n_jobs=1)
1.32 s ± 45.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

In [270]: %timeit ccc(data, n_jobs=2)
771 ms ± 11 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```


## Reproducible research

Below we provide the steps to reproduce all the analyses in the CCC manuscript.

### Setup

To prepare the environment to run the analyses in the manuscript, follow the steps in [environment](environment/).
This will create a conda environment and download the necessary data.
Alternatively, you can use our Docker image (see below).

### Running code

**From command-line.**
First, activate your conda environment and export your settings to environmental variables so non-Python scripts can access them:
```bash
conda activate ccc
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

**From your browser.**
Alternatively, you can start your JupyterLab server by running:

```bash
bash scripts/run_nbs_server.sh
```

Then, go to `http://localhost:8892`, browse the `nbs` folder, and run the notebooks in the specified order.

## Using Docker

You can also run all the steps below using a Docker image instead of a local installation.

```bash
docker pull miltondp/ccc
```

The image only contains the conda environment with the code in this repository so, after pulling the image, you need to download the data as well:

```bash
mkdir -p /tmp/ccc_data

docker run --rm \
  -v "/tmp/ccc_data:/opt/data" \
  --user "$(id -u):$(id -g)" \
  miltondp/ccc \
  /bin/bash -c "python environment/scripts/setup_data.py"
```

The `-v` parameter allows specifying a local directory (`/tmp/ccc_data`) where the data will be downloaded.
If you want to generate the figures and tables for the manuscript, you need to clone the [manuscript repo](https://github.com/greenelab/ccc-manuscript) and pass it with `-v [PATH_TO_MANUSCRIPT_REPO]:/opt/manuscript`.
If you want to change any other setting, you can set environmental variables when running the container; for example, to change the number of cores used to 2: `-e CM_N_JOBS=2`.

You can run notebooks from the command line, for example:

```bash
docker run --rm \
  -v "/tmp/ccc_data:/opt/data" \
  --user "$(id -u):$(id -g)" \
  miltondp/ccc \
  /bin/bash -c "parallel -k --lb --halt 2 -j1 'bash nbs/run_nbs.sh {}' ::: nbs/05_preprocessing/*.ipynb"
```

or start a Jupyter Notebook server with:

```bash
docker run --rm \
  -p 8888:8893 \
  -v "/tmp/ccc_data:/opt/data" \
  --user "$(id -u):$(id -g)" \
  miltondp/ccc
```

FIXME: why do I have a different port here than before?

and access the interface by going to `http://localhost:8888`.
