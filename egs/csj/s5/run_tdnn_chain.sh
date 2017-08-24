#!/bin/bash

# This is modified from swbd/s5c/local/nnet3/run_tdnn.sh
# Tomohiro Tanaka 15/05/2016

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

. ./cmd.sh

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

train_stage=-10
stage=0
common_egs_dir=
reporting_email=
remove_egs=true

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

dir=exp/nnet3/nnet_tdnn
train_set=train_nodup
ali_dir=exp/tri4_ali_nodup
if [ -e data/train_dev ] ;then
    dev_set=train_dev
fi

local/nnet3/run_ivector_common.sh --stage $stage || exit 1;
local/chain/run_tdnn_swbd_mod4csj.sh || exit 1; 

./utils/mkgraph.sh --self-loop-scale 1.0 data/lang_csj_tg exp/chain/tdnn_csj exp/chain/tdnn_csj/graph_csj_tg

for decode_set in eval1 eval2 eval3
do
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --nj 10 --cmd "run.pl"  --online-ivector-dir exp/nnet3/ivectors_${decode_set} exp/chain/tdnn_csj/graph_csj_tg data/${decode_set}_hires exp/chain/tdnn_csj/decode_${decode_set}
done

steps/online/nnet3/prepare_online_decoding.sh  --mfcc-config conf/mfcc_hires.conf data/lang_chain_2y exp/nnet3/extractor exp/chain/tdnn_csj exp/chain/tdnn_csj_online

for decode_set in eval1 eval2 eval3
do
  steps/online/nnet3/decode.sh --nj 10 --cmd "run.pl" --acwt 1.0 --post-decode-acwt 10.0 exp/chain/tdnn_csj/graph_csj_tg data/${decode_set}_hires exp/chain/tdnn_csj_online/decode_${decode_set}
done


exit 0;

