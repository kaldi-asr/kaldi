#!/bin/bash
# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)   

#SBATCH --export=PATH,LD_LIBRARY_PATH
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3


srun fairseq_ltlm/recipes/tuning/start_ddp_train.sh $@ || exit 1

echo "Done"
