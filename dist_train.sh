#!/usr/bin/env bash
export PATH="/nfs/volume-904-4/wangjiabao_i/cuda/cuda-11.6/bin:$HOME/.local/bin/:$HOME/bin:/usr/local/bin:$PATH"
export LD_LIBRARY_PATH="/nfs/volume-904-4/wangjiabao_i/cuda/cuda-11.6/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME="/nfs/volume-904-4/wangjiabao_i/cuda/cuda-11.6"

GPUS=$1
CONFIG=$2
python3 -m torch.distributed.run --nproc_per_node $GPUS train.py --config $CONFIG ${@:3}
