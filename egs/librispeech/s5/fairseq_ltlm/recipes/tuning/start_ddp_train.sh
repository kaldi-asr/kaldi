#!/usr/bin/env bash
# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)
   
echo "hostname: $(hostname)"
 
set | grep SLURM
 
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST 2>&1 | head -n 1)
echo "MASTER_ADDR: $MASTER_ADDR"
export MASTER_PORT=5517
echo "MASTER_PORT: $MASTER_PORT"
 
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
#export NCCL_SOCKET_IFNAME=$(ip a | grep 'inet 10\.16\.1\..*/24' | awk '{print $7}')
#echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
 
#export WORLD_SIZE=$SLURM_NTASKS
echo "WORLD_SIZE: $WORLD_SIZE"
 

. ./fairseq_ltlm/path.sh
 
echo "$@"
python fairseq_ltlm/fairseq/train.py \
		--distributed-port=$MASTER_PORT \
		--distributed-world-size=$WORLD_SIZE \
		$@ || exit 1

