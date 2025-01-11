# Python environment and data download

If you want to run the scripts/notebooks, you need to follow these steps to create a Python environment and download the necessary data.
Although you can create your own Python environment, using our Docker image is much easier.

Keep in mind that although unit tests are automatically run on Linux, macOS and MS Windows, the software is manually tested only on Linux/Ubuntu.

There are four steps:
1. Getting the source code (cloning this repository).
1. Adjusting settings (paths, etc). 
1. Creating a Python environment.
1. Downloading the data to run the analyses.

## Getting the source code

```bash
git clone https://github.com/greenelab/ccc.git
cd ccc/
```

## Adjusting settings

Adjust the paths where the data will be downloaded, the number of CPU cores available to run the code, etc.
You can specify these options using environment variables in your terminal:

 ```bash
 # (optional) Root directory where all data will be downloaded to.
 # Defaults to subfolder 'ccc_gene_expr' under the system's temporary directory.
 export CM_ROOT_DIR=/tmp/ccc_gene_expr

 # (optional) Adjust the number of cores available for general tasks.
 # Defaults to 1 CPU core.
 export CM_N_JOBS=2

 # (optional) Export this variable if you downloaded the manuscript sources
 # and want to generate the figures for it.
 export CM_MANUSCRIPT_DIR=/tmp/manuscript
 ```

or you can change these options in `../libs/ccc/settings.py`

## Creating a Python environment

Now you need a Python environment.
You have two choices: 1) using Docker (the easiest) or 2) creating your own conda environment.

### Using Docker

This is the easiest approach.
First, you need to [install Docker](https://docs.docker.com/get-docker/) for your system.
Then, you have to download our the Docker image for CCC with this command (you might need to add `sudo` at the beginning if your user is not part of the `docker` group):

```bash
docker pull miltondp/ccc
```

The `miltondp/ccc` image only contains the Python environment and all the needed packages and tools to run the code.
You still need to download the data (see below).

### Creating your own conda environment

Follow these steps if you are creating your own conda environment.
Here you'll also export some environment variables like `PYTHONPATH` or the CCC configuration.
Remember that these environment variables need to be present in the terminal where you'll run the analyses/notebooks.

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

1. `cd` into `environment` inside the CCC root folder:

```bash
cd environment/
```

1. Adjust your `PYTHONPATH` variable to include the `libs` directory:

 ```bash
 export PYTHONPATH=`readlink -f ../libs/`:$PYTHONPATH
 ```

 `readlink` might not work on macOS. In that case, simply replace it with
 the absolute path to the `../libs/` folder.

1. Create a conda environment and install main packages:

 ```bash
conda env create --name ccc --file environment-cuda.yml
conda run -n ccc --no-capture-output bash scripts/install_other_packages.sh
 ```

1. Export the entire configuration into environment variables (this is useful for bash scripts and R notebooks so they can also read the configuration):

```bash
eval `python ../libs/ccc/conf.py`
```

## Downloading the data

Move to the root folder of the repository:

```bash
cd ..
```

The command to download all the necessary data is `python environment/scripts/setup_data.py`.
If you are using your own conda environment, then just run that command on the same terminal (so you keep the environment variables you set).
If you are using Docker, then you can run:

```bash
bash scripts/run_docker.sh \
  python environment/scripts/setup_data.py
```

This will download 1.6G GB of data.

The script `scripts/run_docker.sh` automatically reads your settings, mounts the repo and root directories into the Docker container and runs the command you specified within it.


## (Internal) Steps to update the conda environment

These steps are for internal use only, you don't need to run them if you are user.

1. Modify `scripts/environment_base.yml` accordingly (if needed).
1. Run:
 
```bash
conda env create -n ccc -f scripts/environment_base.yml
conda activate ccc
bash scripts/install_other_packages.sh
```

1. Export conda environment:

```bash
conda env export --name ccc --file environment-cuda.yml
```

1. Modify `environment.yml` and leave only manually installed packages (not their dependencies).
