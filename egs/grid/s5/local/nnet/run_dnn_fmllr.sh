#!/bin/bash

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
#                2017  Ruhr-University Bochum (Author: Hendrik Meutzner)
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

#. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

#. ./path.sh ## Source the tools/utils (import the queue.pl)

echo ""
echo "$0"
date

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh ## Source the tools/utils (import the queue.pl)


stage=0 # resume training with --stage=N
srcgmm=tri3b # source GMM model to create alignments
dstdnn=dnn_tri3b_fmmlr # destination DNN model
graphdir=
do_smbr=true
eval_list="devel test"

acwt=0.1

. ./utils/parse_options.sh || exit 1;

if [ $# -ge 1 ]; then
  proc=$1
else
  proc=`pwd`
fi

if [ $# -ge 2 ]; then
  feat=$2
else
  feat=
fi

datadir=$proc/data
data_fmllr=$proc/data-fmllr
expdir=$proc/exp
langdir=$datadir/lang
dict=$datadir/local/dict

if [ -n "$feat" ]; then
  datadir=$datadir/$feat
  expdir=$expdir/$feat
fi

gmmdir=${expdir}/${srcgmm}
if [ -z "$graphdir" ]; then
  graphdir=$gmmdir/graph
fi

if [ $stage -le 0 ]; then
  # Store fMLLR features, so we can train on them easily,

  for x in ${eval_list}; do
    dir=$data_fmllr/$x
    steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
       --transform-dir $gmmdir/decode_${x} \
       $dir $datadir/$x $gmmdir $dir/log $dir/data || exit 1
  done

  # train
  dir=$data_fmllr/train
  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
     --transform-dir ${gmmdir}_ali \
     $dir $datadir/train $gmmdir $dir/log $dir/data || exit 1

  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
fi


if [ $stage -le 1 ]; then
  # Pre-train DBN, i.e. a stack of RBMs
  dir=$expdir/${dstdnn}_pretrain-dbn

  (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log

  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --rbm-iter 3 $data_fmllr/train $dir || exit 1;
fi

dir=$expdir/${dstdnn}_pretrain-dbn_dnn

if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  ali=${gmmdir}_ali

  feature_transform=$expdir/${dstdnn}_pretrain-dbn/final.feature_transform
  dbn=$expdir/${dstdnn}_pretrain-dbn/6.dbn

  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log

  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    $data_fmllr/train_tr90 $data_fmllr/train_cv10 $langdir $ali $ali $dir || exit 1;
fi

if [ $stage -le 3 ]; then
  # Decode (reuse HCLG graph)

  for x in ${eval_list}; do

    local/nnet/compute_logprob.sh --nj 10 --cmd "$decode_cmd" \
      $graphdir $data_fmllr/$x $dir/decode_${x} || exit 1;

    local/nnet/decode_logprob.sh --nj 10 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
      $graphdir $data_fmllr/$x $dir/decode_${x} || exit 1;

  done

fi

if [ $do_smbr = false ]; then
  echo "$0: Skipping sequence training using sMBR criterion."
  exit 0;
fi

# Sequence training using sMBR criterion, we do Stochastic-GD
# with per-utterance updates. We use usually good acwt 0.1
# Lattices are re-generated after 1st epoch, to get faster convergence.
dir=$expdir/${dstdnn}_pretrain-dbn_dnn_smbr
srcdir=$expdir/${dstdnn}_pretrain-dbn_dnn

if [ $stage -le 4 ]; then
  # First we generate lattices and alignments:
  steps/nnet/align.sh --nj 10 --cmd "$train_cmd" \
    $data_fmllr/train $langdir $srcdir ${srcdir}_ali || exit 1;
fi

if [ $stage -le 5 ]; then
  steps/nnet/make_denlats.sh --nj 10 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
    $data_fmllr/train $langdir $srcdir ${srcdir}_denlats || exit 1;
fi

if [ $stage -le 6 ]; then
  # Re-train the DNN by 1 iteration of sMBR
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 5 --acwt $acwt --do-smbr true \
    $data_fmllr/train $langdir $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
fi

if [ $stage -le 7 ]; then
  # Decode (reuse HCLG graph)
  for ITER in 5 4 3 1; do

    for x in ${eval_list}; do

      local/nnet/compute_logprob.sh --nj 10 --cmd "$decode_cmd" \
        --nnet $dir/${ITER}.nnet \
        $graphdir $data_fmllr/${x} $dir/decode_${x}_it${ITER} || exit 1;

      local/nnet/decode_logprob.sh --nj 10 --cmd "$decode_cmd" --config conf/decode_dnn.config \
        --acwt $acwt \
        $graphdir $data_fmllr/${x} $dir/decode_${x}_it${ITER} || exit 1;

    done

  done
fi

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
