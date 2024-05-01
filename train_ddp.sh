#!/usr/bin/env bash
CONFIG=$1

source activate pointocc
python3 -m torch.distributed.launch --nproc_per_node $MLP_WORKER_GPU --master_addr $MLP_WORKER_0_HOST --node_rank $MLP_ROLE_INDEX --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM  train_ddp.py --config $CONFIG