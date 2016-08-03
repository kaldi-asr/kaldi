#!/bin/bash

# 2016 Modified by Takafumi Moriya at Tokyo Institute of Technology
# for Japanese speech recognition using CSJ.

# Copyright 2015  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains a LSTM network on FBANK features.
# The LSTM code comes from Yiayu DU, and Wei Li, thanks!

. ./cmd.sh
. ./path.sh

if [ -e data/train_dev ] ;then
    dev_set=train_dev
fi

train=data-fbank/train_nodup
train_original=data/train_nodup

gmm=exp/tri4

stage=0
. utils/parse_options.sh || exit 1;

# Make the FBANK features
[ ! -e $train ] && if [ $stage -le 0 ]; then
  # evaluation set
    for eval_num in eval1 eval2 eval3 $dev_set ;do
        dir=data-fbank/$eval_num; srcdir=data/$eval_num
        (mkdir -p $dir; cp $srcdir/* $dir; )
        utils/copy_data_dir.sh data/$eval_num $dir || exit 1; rm $dir/{cmvn,feats}.scp
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 $dir $dir/log $dir/data || exit 1;
        steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
    done
    
  # Training set
  utils/copy_data_dir.sh $train_original $train || exit 1; rm $train/{cmvn,feats}.scp
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd -tc 10" \
     $train $train/log $train/data || exit 1;
  steps/compute_cmvn_stats.sh $train $train/log $train/data || exit 1;
  # Split the training set
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $train ${train}_tr90 ${train}_cv10
fi

if [ $stage -le 1 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/lstm4
  ali=${gmm}_ali_nodup

  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --network-type lstm --learn-rate 0.0001 \
      --cmvn-opts "--norm-means=true --norm-vars=true" --feat-type plain --splice 0 \
      --train-opts "--momentum 0.9 --halving-factor 0.5" \
      --train-tool "nnet-train-lstm-streams --num-stream=4 --targets-delay=5" \
      --proto-opts "--num-cells 512 --num-recurrent 200 --num-layers 2 --clip-gradient 5.0" \
    ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir || exit 1;

  # Decode (reuse HCLG graph)
  for eval_num in eval1 eval2 eval3 $dev_set ;do
      steps/nnet/decode.sh --nj 10 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.2 \
    $gmm/graph_csj_tg data-fbank/$eval_num $dir/decode_${eval_num}_csj || exit 1;
  done
fi

# TODO : sequence training,

echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
# We use config parameters of rm resipe.
# TODO : Tuning the parameters.
:<<EOF
=== evaluation set 1 ===
%WER 13.24 [ 3446 / 26028, 372 ins, 803 del, 2271 sub ] exp/lstm4/decode_eval1_csj/wer_11_0.5
=== evaluation set 2 ===
%WER 10.53 [ 2808 / 26661, 376 ins, 436 del, 1996 sub ] exp/lstm4/decode_eval2_csj/wer_11_0.0
=== evaluation set 3 ===
%WER 14.51 [ 2494 / 17189, 402 ins, 381 del, 1711 sub ] exp/lstm4/decode_eval3_csj/wer_12_0.0
EOF
