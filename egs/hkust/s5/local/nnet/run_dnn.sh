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
gmmdir=exp/tri5a
data_fmllr=data-fmllr-tri5a
stage=0 # resume training with --stage=N
# End of config.
. utils/parse_options.sh || exit 1;
#

if [ $stage -le 0 ]; then
  # Store fMLLR features, so we can train on them easily,
  # dev
  dir=$data_fmllr/dev
  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode \
     $dir data/dev $gmmdir $dir/log $dir/data || exit 1
  # train
  dir=$data_fmllr/train
  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
     --transform-dir ${gmmdir}_ali \
     $dir data/train $gmmdir $dir/log $dir/data || exit 1
  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
fi

if [ $stage -le 1 ]; then
  # Pre-train DBN, i.e. a stack of RBMs
  dir=exp/dnn5b_pretrain-dbn
  (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --rbm-iter 1 --nn-depth 4 --hid-dim 2000 \
    $data_fmllr/train $dir || exit 1;
fi

if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/dnn5b_pretrain-dbn_dnn
  ali=${gmmdir}_ali
  feature_transform=exp/dnn5b_pretrain-dbn/final.feature_transform
  dbn=exp/dnn5b_pretrain-dbn/4.dbn
  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    $data_fmllr/train_tr90 $data_fmllr/train_cv10 data/lang $ali $ali $dir || exit 1;
  # Decode with the trigram language model.
  steps/nnet/decode.sh --nj 10 --cmd "$decode_cmd" \
    --config conf/decode_dnn.config --acwt 0.1 \
    $gmmdir/graph $data_fmllr/dev \
    $dir/decode || exit 1;
fi

# Sequence training using sMBR criterion, we do Stochastic-GD 
# with per-utterance updates. We use usually good acwt 0.1
# Lattices are re-generated after 1st epoch, to get faster convergence.
dir=exp/dnn5b_pretrain-dbn_dnn_smbr
srcdir=exp/dnn5b_pretrain-dbn_dnn
acwt=0.1

if [ $stage -le 3 ]; then
  # First we generate lattices and alignments:
  steps/nnet/align.sh --nj 10 --cmd "$train_cmd" \
    $data_fmllr/train data/lang $srcdir ${srcdir}_ali || exit 1;
  steps/nnet/make_denlats.sh --nj 10 --sub-split 20 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --acwt $acwt $data_fmllr/train data/lang $srcdir ${srcdir}_denlats || exit 1;
fi

if [ $stage -le 4 ]; then
  # Re-train the DNN by 1 iteration of sMBR 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 1 --acwt $acwt --do-smbr true \
    $data_fmllr/train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode (reuse HCLG graph)
  for ITER in 1; do
    # Decode with the trigram swbd language model.
    steps/nnet/decode.sh --nj 10 --cmd "$decode_cmd" \
      --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmmdir/graph $data_fmllr/dev \
      $dir/decode || exit 1;
  done 
fi

# Re-generate lattices, run 2 more sMBR iterations
dir=exp/dnn5b_pretrain-dbn_dnn_smbr_i1lats
srcdir=exp/dnn5b_pretrain-dbn_dnn_smbr
acwt=0.0909

if [ $stage -le 5 ]; then
  # First we generate lattices and alignments:
  #steps/nnet/align.sh --nj 10 --cmd "$train_cmd" \
  #  $data_fmllr/train data/lang $srcdir ${srcdir}_ali || exit 1;
  steps/nnet/make_denlats.sh --nj 10 --sub-split 20 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --acwt $acwt $data_fmllr/train data/lang $srcdir ${srcdir}_denlats || exit 1;
fi

if [ $stage -le 6 ]; then
  # Re-train the DNN by 2 iteration of sMBR 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 2 --acwt $acwt --do-smbr true \
    $data_fmllr/train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode (reuse HCLG graph)
  for ITER in 1 2; do
    # Decode with the trigram language model.
    steps/nnet/decode.sh --nj 10 --cmd "$decode_cmd" \
      --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmmdir/graph $data_fmllr/dev \
      $dir/decode || exit 1;
  done 
fi

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
