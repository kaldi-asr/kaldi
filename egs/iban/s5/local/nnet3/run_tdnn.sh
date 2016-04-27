#!/bin/bash

# Combined from from WSJ + RM 

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

. ./cmd.sh


# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=9
train_stage=-10
dir=exp/nnet3/nnet_tdnn_h_sp_4_850_170
. cmd.sh
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
  #  --splice-indexes "-2,-1,0,1,2 -3,1 -5,3 0 0 " \
  #  --splice-indexes "-4,-3,-2,-1,0,1,2,3,4  0  -2,2  0  -4,4 0" \

  steps/nnet3/train_tdnn.sh --stage $train_stage \
    --num-jobs-initial 2 --num-jobs-final 4 \
    --splice-indexes "-4,-3,-2,-1,0,1,2,3,4  0  -2,2  0  -4,4 0 0" \
    --num-epochs 4 \
    --add-layers-period 1 \
    --feat-type raw \
    --online-ivector-dir exp/nnet3/ivectors_train\
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --initial-effective-lrate 0.005 --final-effective-lrate 0.0005 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim 850 \
    --pnorm-output-dim 170 \
    --num-jobs-compute-prior 4\
    data/train_hires data/lang exp/nnet3/tri3b_ali_sp $dir  || exit 1;
fi


if [ $stage -le 10 ]; then
  # this does offline decoding that should give the same results as the real
  # online decoding.
  graph_dir=exp/tri3b/graph
  # use already-built graphs.
    steps/nnet3/decode.sh --nj 6 --cmd "$decode_cmd" \
        --online-ivector-dir exp/nnet3/ivectors_dev --iter final\
       $graph_dir data/dev_hires $dir/decode_dev || exit 1;
  
fi

if [ $stage -le 11 ]; then
   steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" \
     data/lang_test/ data/lang_big/ data/dev \
    ${dir}/decode_dev ${dir}/decode_dev.rescored
fi

exit 0;

