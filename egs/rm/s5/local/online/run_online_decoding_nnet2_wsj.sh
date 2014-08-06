#!/bin/bash

# This script assumes you have previously run the WSJ example script including
# the optional part local/online/run_online_decoding_nnet2.sh.  It builds a
# neural net for online decoding on top of the network we previously trained on
# WSJ, by keeping everything but the last layer of that network and then
# training just the last layer on our data.


# !!NOT FINISHED!!
stage=0


train_stage=-10
use_gpu=true
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  parallel_opts="-l gpu=1" 
  num_threads=1
  minibatch_size=512
  dir=exp/nnet2_online_wsj/nnet_gpu
  srcdir=../../wsj/s5/exp/nnet2_online/nnet_gpu_online
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="-pe smp $num_threads" 
  dir=exp/nnet2_online_wsj/nnet
  srcdir=../../wsj/s5/exp/nnet2_online/nnet_online
fi

