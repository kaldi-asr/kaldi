#!/bin/bash

# this script acts like a wrapper of a bash command that uses GPUs.
#
# it sets the CUDA_VISIBLE_DEVICES variable so that it limit the number of GPUs
# used for programs. It is neccesary for the grid we run a program like
# TensorFlow which automatically grabs all resources available on the system

num_gpus=1

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

CUDA_VISIBLE_DEVICES=
num_total_gpus=`nvidia-smi -L | wc -l | awk '{print $1}'`
num_gpus_assigned=0

for i in `seq 0 $[$num_total_gpus-1]`; do
# going over all GPus and check if idle and add to the list
  idle=`nvidia-smi -i $i | grep "No running processes found" | wc -l | awk '{print $1}'`
  [ $idle -eq 1 ] && CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}$i, && num_gpus_assigned=$[$num_gpus_assigned+1]

# once we have enough GPUs break out of the loop
  [ $num_gpus_assigned -eq $num_gpus ] && break
done

[ $num_gpus_assigned -ne $num_gpus ] && echo Could not find enough idle GPUs && exit -1

CUDA_VISIBLE_DEVICES=`echo $CUDA_VISIBLE_DEVICES | sed "s=,$==g"`
echo Running the job on GPU\(s\) $CUDA_VISIBLE_DEVICES

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $@
