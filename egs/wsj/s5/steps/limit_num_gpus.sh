#!/bin/bash

# This script functions as a wrapper of a bash command that uses GPUs.
#
# It sets the CUDA_VISIBLE_DEVICES variable so that it limits the number of GPUs
# used for programs. It is neccesary for running a job on the grid if the job
# would automatically grabs all resources available on the system, e.g. a
# TensorFlow program.

num_gpus=1

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

CUDA_VISIBLE_DEVICES=
num_total_gpus=`nvidia-smi -L | wc -l | awk '{print $1}'`
num_gpus_assigned=0

for i in `seq 0 $[$num_total_gpus-1]`; do
# going over all GPUs and check if it is idle, and add to the list if yes
  idle=`nvidia-smi -i $i | grep "No running processes found" | wc -l | awk '{print $1}'`
  [ $idle -eq 1 ] && CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}$i, && num_gpus_assigned=$[$num_gpus_assigned+1]

# once we have enough GPUs, break out of the loop
  [ $num_gpus_assigned -eq $num_gpus ] && break
done

[ $num_gpus_assigned -ne $num_gpus ] && echo Could not find enough idle GPUs && exit -1

CUDA_VISIBLE_DEVICES=`echo $CUDA_VISIBLE_DEVICES | sed "s=,$==g"`
echo Running the job on GPU\(s\) $CUDA_VISIBLE_DEVICES

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $@
