conda activate ccc
export CODE_DIR=/home/haoyu/_database/projs/ccc-gpu
export PYTHONPATH=`readlink -f ./libs/`:$PYTHONPATH
eval `python ./libs/ccc/conf.py`
