#!/usr/bin/env bash
CONFIG=$1

source activate pointocc
python3 -m torch.distributed.run --nproc_per_node 8 train.py --config $CONFIG