#!/bin/bash

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
#                2014  Guoguo Chen
# Apache 2.0

# This example script trains a DNN on top of fMLLR features. 
# The training is done in 3 stages,
#
# 1) RBM pre-training:
#    in this unsupervised stage we train stack of RBMs, 
#    a good starting point for frame cross-entropy trainig.
# 2) frame cross-entropy training:
#    the objective is to classify frames to correct pdfs.
# 3) sequence-training optimizing sMBR: 
#    the objective is to emphasize state-sequences with better 
#    frame accuracy w.r.t. reference alignment.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

# Config:
gmmdir=exp/tri4
data_fmllr=data-fmllr-tri4
stage=0 # resume training with --stage=N
has_fisher=true
# End of config.
. utils/parse_options.sh || exit 1;
#

set -euxo pipefail

if [ $stage -le 0 ]; then
  # Store fMLLR features, so we can train on them easily,
  # eval2000
  dir=$data_fmllr/eval2000
  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode_eval2000_sw1_tg \
     $dir data/eval2000 $gmmdir $dir/log $dir/data
  # train
  dir=$data_fmllr/train_nodup
  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
     --transform-dir ${gmmdir}_ali_nodup \
     $dir data/train_nodup $gmmdir $dir/log $dir/data
  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10
fi

if [ $stage -le 1 ]; then
  # Pre-train DBN, i.e. a stack of RBMs
  dir=exp/dnn5b_pretrain-dbn
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --rbm-iter 1 $data_fmllr/train_nodup $dir
fi

if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/dnn5b_pretrain-dbn_dnn
  ali=${gmmdir}_ali_nodup
  feature_transform=exp/dnn5b_pretrain-dbn/final.feature_transform
  dbn=exp/dnn5b_pretrain-dbn/6.dbn
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    $data_fmllr/train_nodup_tr90 $data_fmllr/train_nodup_cv10 data/lang $ali $ali $dir
  # Decode with the trigram swbd language model.
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" \
    --config conf/decode_dnn.config --acwt 0.08333 \
    $gmmdir/graph_sw1_tg $data_fmllr/eval2000 \
    $dir/decode_eval2000_sw1_tg
  if $has_fisher; then
    # Rescore with the 4gram swbd+fisher language model.
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_sw1_{tg,fsh_fg} data/eval2000 \
      $dir/decode_eval2000_sw1_{tg,fsh_fg}
  fi
fi


# Sequence training using sMBR criterion, we do Stochastic-GD 
# with per-utterance updates. The typical acwt value is around 0.1
dir=exp/dnn5b_pretrain-dbn_dnn_smbr
srcdir=exp/dnn5b_pretrain-dbn_dnn
acwt=0.0909

if [ $stage -le 3 ]; then
  # First we generate lattices and alignments:
  steps/nnet/align.sh --nj 250 --cmd "$train_cmd" \
    $data_fmllr/train_nodup data/lang $srcdir ${srcdir}_ali
  steps/nnet/make_denlats.sh --nj 10 --sub-split 100 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --acwt $acwt $data_fmllr/train_nodup data/lang $srcdir ${srcdir}_denlats
fi

if [ $stage -le 4 ]; then
  # Re-train the DNN by 1 iteration of sMBR 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 4 --acwt $acwt --do-smbr true \
    $data_fmllr/train_nodup data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir
  # Decode (reuse HCLG graph)
  for ITER in 4 3 2 1; do
    # Decode with the trigram swbd language model.
    steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" \
      --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmmdir/graph_sw1_tg $data_fmllr/eval2000 \
      $dir/decode_eval2000_sw1_tg_it$ITER
    if $has_fisher; then
      # Rescore with the 4gram swbd+fisher language model.
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_sw1_{tg,fsh_fg} data/eval2000 \
        $dir/decode_eval2000_sw1_{tg,fsh_fg}_it$ITER
    fi
  done 
fi

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
