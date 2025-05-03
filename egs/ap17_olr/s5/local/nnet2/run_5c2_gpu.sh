#!/bin/bash

# This is neural net training on top of adapted 40-dimensional features.
# This is an alternative to the run_5c_gpu.sh that will train faster if you
# have 8 gpus because it uses more jobs, but the results are slightly worse.
# [note: possibly we could raise the learning rate and match the run_5c_gpu.sh
# results.]


train_stage=-100
temp_dir=  # e.g. --temp-dir /export/m1-02/dpovey/kaldi-dan2/egs/wsj/s5/
parallel_opts="--gpu 1"  # This is suitable for the CLSP network, you'll likely have to change it.
dir=exp/nnet5c2_gpu

# Note: since we multiplied the num-jobs by 1/4, we halved the
# learning rate, relative to run_5c.sh

. ././cmd.sh
. ./path.sh
! cuda-compiled && cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
. utils/parse_options.sh

( 

  if [ ! -z "$temp_dir" ] && [ ! -e $dir/egs ]; then
    mkdir -p $dir
    mkdir -p $temp_dir/$dir/egs
    ln -s $temp_dir/$dir/egs $dir/
  fi

  steps/nnet2/train_tanh.sh \
   --num-jobs-nnet 8 --num-threads 1 --parallel-opts "$parallel_opts" \
   --mix-up 8000 \
   --initial-learning-rate 0.0075 --final-learning-rate 0.00075 \
   --num-hidden-layers 4 --hidden-layer-dim 1024 \
   --cmd "$decode_cmd" \
    data/train_si284 data/lang exp/tri4b_ali_si284 $dir || exit 1
  
  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 10 \
    --transform-dir exp/tri4b/decode_bd_tgpr_dev93 \
     exp/tri4b/graph_bd_tgpr data/test_dev93 $dir/decode_bd_tgpr_dev93

  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 8 \
    --transform-dir exp/tri4b/decode_bd_tgpr_eval92 \
     exp/tri4b/graph_bd_tgpr data/test_eval92 $dir/decode_bd_tgpr_eval92
)

