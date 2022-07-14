# Clustermatch Correlation Coefficient (CCC)

[![Code tests](https://github.com/greenelab/ccc/actions/workflows/pytest.yaml/badge.svg)](https://github.com/greenelab/ccc/actions/workflows/pytest.yaml)
[![codecov](https://codecov.io/gh/greenelab/ccc/branch/main/graph/badge.svg?token=QNK6O3Y1VF)](https://codecov.io/gh/greenelab/ccc)
[![bioRxiv Manuscript](https://img.shields.io/badge/manuscript-bioRxiv-blue.svg)](https://doi.org/10.1101/2022.06.15.496326)
[![HTML Manuscript](https://img.shields.io/badge/manuscript-HTML-blue.svg)](https://greenelab.github.io/ccc-manuscript/)

## Overview

The Clustermatch Correlation Coefficient (CCC) is a highly-efficient, next-generation not-only-linear correlation coefficient that can work on numerical and categorical data types.
This repository contains the code of CCC and instructions to install and use it.
It also has all the scripts/notebooks to run the analyses associated with the [manuscript](https://github.com/greenelab/ccc-manuscript), where we applied CCC on gene expression data.

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

When computing the correlation coefficient on a pair of features, CCC supports `numpy.array` or `pandas.Series`.
Missing values (`NaN`) are not currently supported, so you have to either remove or impute them.
Below there is an example with numerical data (you can copy/paste the entire lines below including `In [...]`):

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
[As we show in the manuscript](https://greenelab.github.io/ccc-manuscript/#the-ccc-reveals-linear-and-nonlinear-patterns-in-human-transcriptomic-data), the distribution of CCC values is much more skewed than other coefficients like Pearson's or Spearman's.
A comparison between these coefficients should account for that.

You can also mix numerical and categorical data:

```python
In [10]: categories = np.array(["blue", "red", "green", "yellow"])
In [11]: categorical_random_feature1 = np.random.choice(categories, size=1000)
In [12]: categorical_random_feature2 = np.random.choice(categories, size=1000)
In [13]: categorical_random_feature2[:10]
Out[13]: 
array(['yellow', 'red', 'red', 'yellow', 'blue', 'blue', 'red', 'yellow',
       'green', 'blue'], dtype='<U6')

In [14]: ccc(categorical_random_feature1, categorical_random_feature2)
Out[14]: 0.0009263483455638076

In [15]: ccc(random_feature1, categorical_random_feature2)
Out[15]: 0.0015123522641692117
```

The first argument of `ccc` could also be a matrix, either as a `numpy.array` (features are in rows and objects in columns) or as a `pandas.DataFrame` (objects are in rows and features in columns).
In this case, `ccc` will compute the pairwise correlation across all features:

```python
In [16]: # with a numpy.array
In [17]: data = np.random.rand(10, 1000)
In [18]: c = ccc(data)
In [19]: c.shape
Out[19]: (45,)
In [20]: c[:10]
Out[20]: 
array([0.00404461, 0.00185342, 0.00248847, 0.00232761, 0.00260786,
       0.00121495, 0.00227679, 0.00099051, 0.00313611, 0.00323936])

In [21]: # with a pandas.DataFrame
In [22]: data_df = pd.DataFrame(data.T)
In [23]: c = ccc(data_df)
In [24]: c.shape
Out[24]: (45,)
In [25]: c[:10]
Out[25]: 
array([0.00404461, 0.00185342, 0.00248847, 0.00232761, 0.00260786,
       0.00121495, 0.00227679, 0.00099051, 0.00313611, 0.00323936])
```

If your data has a mix of numerical and categorical features, it's better to work on a `pandas.DataFrame`.
As an example, we load the [Titanic dataset](https://www.kaggle.com/c/titanic/data) (from [seaborn](https://github.com/mwaskom/seaborn-data/)'s repository):

```python
In [26]: titanic_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/raw/titanic.csv"
In [27]: titanic_df = pd.read_csv(titanic_url)
In [28]: titanic_df.shape
Out[28]: (891, 11)
In [29]: titanic_df.head()
Out[29]: 
   survived  pclass                                               name     sex   age  sibsp  parch            ticket     fare cabin embarked
0         0       3                            Braund, Mr. Owen Harris    male  22.0      1      0         A/5 21171   7.2500   NaN        S
1         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0          PC 17599  71.2833   C85        C
2         1       3                             Heikkinen, Miss. Laina  female  26.0      0      0  STON/O2. 3101282   7.9250   NaN        S
3         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0            113803  53.1000  C123        S
4         0       3                           Allen, Mr. William Henry    male  35.0      0      0            373450   8.0500   NaN        S
```

The Titanic dataset has missing values:

```python
In [30]: titanic_df.isna().sum()
Out[30]: 
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
In [31]: titanic_df = titanic_df.dropna(subset=["embarked"]).dropna(axis=1)
In [32]: titanic_df.shape
Out[32]: (889, 9)
```

Now we can run CCC on the dataset and get a correlation matrix across features:

```python
In [33]: ccc_corrs = ccc(titanic_df)

In [34]: from scipy.spatial.distance import squareform
In [35]: ccc_corrs = squareform(ccc_corrs)
In [36]: np.fill_diagonal(ccc_corrs, 1.0)
In [37]: ccc_corrs = pd.DataFrame(ccc_corrs, index=titanic_df.columns.tolist(), columns=titanic_df.columns.tolist())
In [38]: ccc_corrs.shape
Out[38]: (9, 9)
In [39]: with pd.option_context('display.float_format', '{:,.2f}'.format): display(ccc_corrs)
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

The `ccc` function also has a `n_jobs` parameter that allows to control the number of CPU cores used.
Parallelization works _across_ variable pairs (if a matrix/dataframe is provided) or by internally distributing computation of a single variable pair.

In the first example below, we compute the pairwise correlation between 20 features across 1000 objects:

```python
In [40]: data = np.random.rand(20, 1000)

In [41]: %timeit ccc(data, n_jobs=1)
1.32 s ± 45.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

In [42]: %timeit ccc(data, n_jobs=2)
771 ms ± 11 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

And here we parallelize the computation of a single variable pair with thousands of objects:

```python
In [43]: x = np.random.normal(size=100000)

In [44]: y = np.random.normal(size=100000)

In [45]: %timeit ccc(x, y, n_jobs=1)
956 ms ± 3.47 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

In [46]: %timeit ccc(x, y, n_jobs=2)
559 ms ± 5.82 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```


## Reproducible research

Below we provide the steps to reproduce all the analyses in the CCC manuscript.
All the analyses are written as Jupyter notebooks and stored in the folder `nbs/`.
All notebooks are organized by directories, such as `01_preprocessing`, with file names that indicate the order in which they should be run.

Here you have the option to create
1) a **minimal environment** to run analyses within your browser, or
2) a **full environment** ready to run all the analyses in your browser or from the command-line (this might be helpful if you plan to launch long-running jobs).

### Setup of a minimal environment

To set up a minimal environment, you first need to [install Docker](https://docs.docker.com/get-docker/) for your system.
Then, open a terminal and run this code:

```bash
# pulls/downloads the Docker image with the environment and source code
sudo docker pull miltondp/ccc

# specify a base directory to store data and code
export BASE_FOLDER="/tmp/ccc"
mkdir -p ${BASE_FOLDER}

# specify a directory in your computer where data and results will be stored
export DATA_FOLDER="${BASE_FOLDER}/data"
mkdir -p ${DATA_FOLDER}

# specify a directory where the source code is
export CODE_FOLDER="${BASE_FOLDER}/code"
git clone https://github.com/greenelab/ccc.git ${CODE_FOLDER}

# download the necessary data (GTEx, etc)
docker run --rm \
  -v "${DATA_FOLDER}:/opt/data" \
  -v "${CODE_FOLDER}:/opt/code" \
  --user "$(id -u):$(id -g)" \
  miltondp/ccc \
  /bin/bash -c "python environment/scripts/setup_data.py"

# run JupyterLab server
docker run --rm \
  -p 8893:8893 \
  -v "${DATA_FOLDER}:/opt/data" \
  -v "${CODE_FOLDER}:/opt/code" \
  --user "$(id -u):$(id -g)" \
  miltondp/ccc 
```

Then open your browser and navigate to http://127.0.0.1:8893/.
With the Jupyter interface, you should open the folder `nbs/`, and then open and run the notebooks in order.
For example, you must start with all notebooks in `05_preprocessing` first (run them in order too), then `10_compute_correlations`, etc.
All data and results will be saved in folder `DATA_FOLDER`, and the code that you change will be saved in your computer under folder `CODE_FOLDER`.
And that's it, you should be able to run all the analyses within your browser.

### Setup of a full environment

To set up a full environment, please follow the steps in [environment](environment/).
After completing those steps, you'll have the source code in this repository, a Python environment (either using a Docker image or creating your own conda environment) and the necessary data to run the analyses.

Once the full environment is set up, then you have two options to run the analyses/notebooks:
1. **Using the browser**:
   1. Start the JupyterLab server. 
   1. Use your browser to open and run the notebooks.
1. **Using the command-line**:
   1. Open a terminal.
   1. Run the notebooks from using [papermill](https://papermill.readthedocs.io/en/latest/).

**Using the browser.**
This option is the standard one and it is likely the one you want to use.
You can run each cell in the notebook, see the output, and change the code if you want.
For example, let's say you want to run the data preprocessing notebooks.
First, you need to start the JupyterLab server.
For this, you can run one of the commands below depending on whether you are using your own conda environment or Docker:

```bash
# if you're using your own conda environment
bash scripts/run_nbs_server.sh

# if you're using Docker
# this will internally run 'bash scripts/run_nbs_server.sh'
bash scripts/run_docker.sh
```

and then go to http://127.0.0.1:8893/ and browse to `nbs/05_preprocessing`.
Then you need to run each notebook in order.

**Using the command-line.**
This is an alternative approach to run notebooks in a more systematic way.
You'll likely not use this option unless you want to run your own analyses in a cluster, for instance.
Here we use the terminal and a tool called `papermill` to run each notebook and write back the results (like figures, etc).
You can see some output in the terminal when it's running, and once finished, you can start the JupyterLab server and open the notebook to see the results.

Using as example the same preprocessing notebooks, you can run all the preprocessing notebooks in order:

```bash
# if you're using your own conda environment
#   requires GNU Parallel: https://www.gnu.org/software/parallel/
#   To install in Ubuntu: apt install parallel
parallel \
  -k \
  --lb \
  --halt 2 \
  -j1 \
  'bash nbs/run_nbs.sh {}' ::: nbs/05_preprocessing/*.ipynb

# if you're using Docker
#   GNU Parallel is already included in the Docker image,
#   so no need to install
bash scripts/run_docker.sh \
  parallel \
    -k \
    --lb \
    --halt 2 \
    -j1 \
    'bash nbs/run_nbs.sh {}' ::: nbs/05_preprocessing/*.ipynb
```

Any command that you add after `scripts/run_docker.sh` will be run inside the Docker container.
