conda activate ccc
export PYTHONPATH=`readlink -f ./libs/`:$PYTHONPATH
eval `python ./libs/ccc/conf.py`
