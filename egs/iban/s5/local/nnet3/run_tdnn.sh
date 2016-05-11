#!/bin/bash

# Combined from from WSJ + RM

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

. ./cmd.sh


# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=1
train_stage=-10
dir=exp/nnet3/tdnn_1

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

local/nnet3/run_ivector_common.sh --stage $stage || exit 1;

if [ $stage -le 9 ]; then
  echo "$0: creating neural net configs";

  # create the config files for nnet initialization
  python steps/nnet3/tdnn/make_configs.py  \
    --feat-dir data/train_hires \
    --ivector-dir exp/nnet3/ivectors_train \
    --ali-dir exp/nnet3/tri3b_ali_sp \
    --relu-dim 256 \
    --splice-indexes=" -2,-1,0,1,2  -1,0,1  -1,0,1  -1,0,1  -1,0,1 -1,0,1 0 " \
    --use-presoftmax-prior-scale true \
   $dir/configs || exit 1;
fi



if [ $stage -le 10 ]; then

  steps/nnet3/train_dnn.py --stage $train_stage \
    --cmd="$decode_cmd" \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 4 \
    --trainer.num-epochs 4 \
    --trainer.add-layers-period 1 \
    --feat.online-ivector-dir exp/nnet3/ivectors_train\
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --trainer.num-epochs 2 \
    --trainer.optimization.initial-effective-lrate 0.005 \
    --trainer.optimization.final-effective-lrate 0.0005 \
    --trainer.samples-per-iter 120000 \
    --cleanup.preserve-model-interval 10 \
    --feat-dir data/train_hires \
    --ali-dir exp/nnet3/tri3b_ali_sp \
    --lang data/lang \
    --dir=$dir  || exit 1;
fi


if [ $stage -le 11 ]; then
  # this does offline decoding that should give the same results as the real
  # online decoding.
  graph_dir=exp/tri3b/graph
  # use already-built graphs.
    steps/nnet3/decode.sh --nj 6 --cmd "$decode_cmd" \
        --online-ivector-dir exp/nnet3/ivectors_dev --iter final\
       $graph_dir data/dev_hires $dir/decode_dev || exit 1;

fi

if [ $stage -le 12 ]; then
   steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" \
     data/lang_test/ data/lang_big/ data/dev \
    ${dir}/decode_dev ${dir}/decode_dev.rescored
fi

exit 0;

