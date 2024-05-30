#!/usr/bin/env bash

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=0
train_stage=-10
dir=exp/nnet3/nnet_tdnn_a
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

if [ $stage -le 8 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/eca-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  # Note that the alignments used come from the pnorm-ensemble model
  # If you choose to skip ensemble training (which is slow), use the best
  # fmllr alignments available (tri4a)
  steps/nnet3/train_tdnn.sh --stage $train_stage \
    --num-epochs 8 --num-jobs-initial 2 --num-jobs-final 14 \
    --splice-indexes "-4,-3,-2,-1,0,1,2,3,4  0  -2,2  0  -4,4 0" \
    --feat-type raw \
    --online-ivector-dir exp/nnet3/ivectors_train \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --initial-effective-lrate 0.005 --final-effective-lrate 0.0005 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim 2000 \
    --pnorm-output-dim 250 \
    data/train_hires data/lang exp/tri5a_ali $dir  || exit 1;
fi


if [ $stage -le 9 ]; then
  # this does offline decoding that should give the same results as the real
  # online decoding.
  graph_dir=exp/tri5a/graph
  # use already-built graphs.
  for data in dev test sup h5; do
    steps/nnet3/decode.sh --nj 8 --cmd "$decode_cmd" \
        --online-ivector-dir exp/nnet3/ivectors_${data} \
       $graph_dir data/${data}_hires $dir/decode_${data} || exit 1;
  done
fi

