#!/bin/bash
   
#SBATCH --export=PATH,LD_LIBRARY_PATH
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3
#SBATCH --partition=asr-2080ti
#SBATCH --exclude=s1



# echo "$@"
srun fairseq_ltlm/recipes/tuning/start_ddp_train.sh $@ || exit 1

echo "Done"
