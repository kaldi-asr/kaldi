#!/bin/bash

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
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
gmmdir=exp/tri4b
data_fmllr=data-fmllr-tri4b
stage=0 # resume training with --stage=N
# End of config.
. utils/parse_options.sh || exit 1;
#

if [ $stage -le 0 ]; then
  # Store fMLLR features, so we can train on them easily,
  # test_dev93
  dir=$data_fmllr/test_dev93
  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode_bd_tgpr_dev93 \
     $dir data/test_dev93 $gmmdir $dir/log $dir/data || exit 1
  # test_eval92
  dir=$data_fmllr/test_eval92
  steps/nnet/make_fmllr_feats.sh --nj 8 --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode_bd_tgpr_eval92 \
     $dir data/test_eval92 $gmmdir $dir/log $dir/data || exit 1
  # train
  dir=$data_fmllr/train_si284
  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
     --transform-dir ${gmmdir}_ali_si284 \
     $dir data/train_si284 $gmmdir $dir/log $dir/data || exit 1
  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
fi

if [ $stage -le 1 ]; then
  # Pre-train DBN, i.e. a stack of RBMs
  dir=exp/dnn5b_pretrain-dbn
  (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --rbm-iter 3 $data_fmllr/train_si284 $dir || exit 1;
fi

if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/dnn5b_pretrain-dbn_dnn
  ali=${gmmdir}_ali_si284
  feature_transform=exp/dnn5b_pretrain-dbn/final.feature_transform
  dbn=exp/dnn5b_pretrain-dbn/6.dbn
  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    $data_fmllr/train_si284_tr90 $data_fmllr/train_si284_cv10 data/lang $ali $ali $dir || exit 1;
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 10 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmmdir/graph_bd_tgpr $data_fmllr/test_dev93 $dir/decode_bd_tgpr_dev93 || exit 1;
  steps/nnet/decode.sh --nj 8 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmmdir/graph_bd_tgpr $data_fmllr/test_eval92 $dir/decode_bd_tgpr_eval92 || exit 1;
fi


# Sequence training using sMBR criterion, we do Stochastic-GD 
# with per-utterance updates. We use usually good acwt 0.1
# Lattices are re-generated after 1st epoch, to get faster convergence.
dir=exp/dnn5b_pretrain-dbn_dnn_smbr
srcdir=exp/dnn5b_pretrain-dbn_dnn
acwt=0.1

if [ $stage -le 3 ]; then
  # First we generate lattices and alignments:
  steps/nnet/align.sh --nj 100 --cmd "$train_cmd" \
    $data_fmllr/train_si284 data/lang $srcdir ${srcdir}_ali || exit 1;
  steps/nnet/make_denlats.sh --nj 100 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
    $data_fmllr/train_si284 data/lang $srcdir ${srcdir}_denlats || exit 1;
fi

if [ $stage -le 4 ]; then
  # Re-train the DNN by 1 iteration of sMBR 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 5 --acwt $acwt --do-smbr true \
    $data_fmllr/train_si284 data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode (reuse HCLG graph)
  for ITER in 5 4 3 1; do
    steps/nnet/decode.sh --nj 10 --cmd "$decode_cmd" --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmmdir/graph_bd_tgpr $data_fmllr/test_dev93 $dir/decode_bd_tgpr_dev93_it${ITER} || exit 1;
    steps/nnet/decode.sh --nj 8 --cmd "$decode_cmd" --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmmdir/graph_bd_tgpr $data_fmllr/test_eval92 $dir/decode_bd_tgpr_eval92_it${ITER} || exit 1;
  done 
fi

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
