#!/usr/bin/env bash

# This script functions as a wrapper of a bash command that uses GPUs.
#
# It sets the CUDA_VISIBLE_DEVICES variable so that it limits the number of GPUs
# used for programs. It is neccesary for running a job on the grid if the job
# would automatically grabs all resources available on the system, e.g. a
# TensorFlow program.

num_gpus=1 # this variable indicates how many GPUs we will allow the command
           # passed to this script will run on. We achieve this by setting the
           # CUDA_VISIBLE_DEVICES variable
set -e

if [ "$1" == "--num-gpus" ]; then
  num_gpus=$2
  shift
  shift
fi

if ! printf "%d" "$num_gpus" >/dev/null || [ $num_gpus -le -1 ]; then
  echo $0: Must pass a positive interger or 0 after --num-gpus
  echo e.g. $0 --num-gpus 2 local/tfrnnlm/run_lstm.sh
  exit 1
fi

if [ $# -eq 0 ]; then
  echo "Usage:  $0 [--num-gpus <num-gpus>] <command> [<arg1>...]"
  echo "Runs <command> with args after setting CUDA_VISIBLE_DEVICES to "
  echo "make sure exactly <num-gpus> GPUs are visible (default: 1)."
  exit 1
fi

CUDA_VISIBLE_DEVICES=
num_total_gpus=`nvidia-smi -L | wc -l`
num_gpus_assigned=0

if [ $num_gpus -eq 0 ] ; then
    echo "$0: Running the job on CPU. Disabling submitting to gpu"
    export CUDA_VISIBLE_DEVICES=""
else
    for i in `seq 0 $[$num_total_gpus-1]`; do
    # going over all GPUs and check if it is idle, and add to the list if yes
      if nvidia-smi -i $i | grep "No running processes found" >/dev/null; then
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}$i, && num_gpus_assigned=$[$num_gpus_assigned+1]
      fi
    # once we have enough GPUs, break out of the loop
      [ $num_gpus_assigned -eq $num_gpus ] && break
    done

    [ $num_gpus_assigned -ne $num_gpus ] && echo Could not find enough idle GPUs && exit 1

    export CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | sed "s=,$==g")

    echo "$0: Running the job on GPU(s) $CUDA_VISIBLE_DEVICES"
fi

"$@"
