#!/bin/bash

# this (local/nnet2/run_6c_gpu.sh) trains a p-norm neural network on top of
# the SAT system in 5a.
# It uses the _fast.sh version of the script, which is faster than the old
# one, and also the --first-component-power 0.5 option, which we believe 
# improves results (we're waiting for the numbers though).


dir=nnet6c5_gpu
train_stage=-10

. ./cmd.sh
. ./path.sh
! cuda-compiled && cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF


. utils/parse_options.sh
parallel_opts="-l gpu=1"  # This is suitable for the CLSP network, you'll likely have to change it.

( 
  if [ "$USER" == dpovey ]; then
     # spread the egs over various machines. 
     utils/create_split_dir.pl /export/b0{1,2,3,4}/dpovey/kaldi-pure/egs/fisher_english_s5/exp/nnet6c_gpu/egs exp/$dir/egs/storage
  fi

  if [ ! -f exp/$dir/final.mdl ]; then

    steps/nnet2/train_pnorm_fast.sh --stage $train_stage --num-epochs 8  \
      --first-component-power 0.5 \
      --egs-dir exp/nnet6c3_gpu/egs \
      --num-epochs-extra 4 \
      --samples-per-iter 400000 \
      --io-opts "-tc 10" \
      --num-jobs-nnet 8 --num-threads 1 \
      --minibatch-size 512 --parallel-opts "$parallel_opts" \
      --mix-up 15000 \
      --initial-learning-rate 0.08 --final-learning-rate 0.008 \
      --num-hidden-layers 5 \
      --pnorm-input-dim 5000 \
      --pnorm-output-dim 500 \
      --cmd "$decode_cmd" \
      data/train data/lang exp/tri5a exp/$dir || exit 1;
  fi

   steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 25 \
     --config conf/decode.config --transform-dir exp/tri5a/decode_dev \
      exp/tri5a/graph data/dev exp/$dir/decode_dev &

)

