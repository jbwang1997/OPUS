#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
python3 -m torch.distributed.run --nproc_per_node $GPUS train.py --config $CONFIG ${@:3}
