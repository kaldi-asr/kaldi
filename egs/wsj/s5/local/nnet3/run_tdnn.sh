#!/bin/bash

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

. cmd.sh


# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=0
train_stage=-10
dir=exp/nnet3/nnet_ms_a
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

local/online/run_ivector_common.sh --stage $stage || exit 1;

if [ $stage -le 8 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/wsj-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi


  steps/nnet3/train_tdnn.sh --stage $train_stage \
    --num-epochs 8 --num-jobs-initial 2 --num-jobs-final 14 \
    --num-hidden-layers 4 \
    --splice-indexes "-4,-3,-2,-1,0,1,2,3,4  0  -2,2  0  -4,4 0" \
    --feat-type raw \
    --online-ivector-dir exp/nnet3/ivectors_train_si284 \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --io-opts "-tc 12" \
    --initial-effective-lrate 0.005 --final-effective-lrate 0.0005 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim 2000 \
    --pnorm-output-dim 250 \
    --mix-up 12000 \
    data/train_si284_hires data/lang exp/tri4b_ali_si284 $dir  || exit 1;
fi


if [ $stage -le 9 ]; then
  # this does offline decoding that should give the same results as the real
  # online decoding.
  for lm_suffix in tgpr bd_tgpr; do
    graph_dir=exp/tri4b/graph_${lm_suffix}
    # use already-built graphs.
    for year in eval92 dev93; do
      steps/nnet3/decode.sh --nj 8 --cmd "$decode_cmd" \
          --online-ivector-dir exp/nnet2_online/ivectors_test_$year \
         $graph_dir data/test_${year}_hires $dir/decode_${lm_suffix}_${year} || exit 1;
    done
  done
fi

