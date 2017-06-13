#!/bin/bash -u

. ./cmd.sh
. ./path.sh

# DNN training. This script is based on egs/swbd/s5b/local/run_dnn.sh
# Shinji Watanabe, Karel Vesely,

# Config:
nj=80
nj_decode=30
stage=0 # resume training with --stage=N
. utils/parse_options.sh || exit 1;
#

if [ $# -ne 1 ]; then
  printf "\nUSAGE: %s [opts] <mic condition(ihm|sdm|mdm)>\n\n" `basename $0`
  exit 1;
fi
mic=$1

gmmdir=exp/$mic/tri4a
data_fmllr=data_${mic}-fmllr-tri4

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7
graph_dir=$gmmdir/graph_${LM}

# Set bash to 'debug' mode, it will exit on : 
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euxo pipefail

# Store fMLLR features, so we can train on them easily,
if [ $stage -le 0 ]; then
  # eval
  dir=$data_fmllr/$mic/eval
  steps/nnet/make_fmllr_feats.sh --nj 15 --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode_eval_${LM} \
     $dir data/$mic/eval $gmmdir $dir/log $dir/data
  # dev
  dir=$data_fmllr/$mic/dev
  steps/nnet/make_fmllr_feats.sh --nj 15 --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode_dev_${LM} \
     $dir data/$mic/dev $gmmdir $dir/log $dir/data
  # train
  dir=$data_fmllr/$mic/train
  steps/nnet/make_fmllr_feats.sh --nj 15 --cmd "$train_cmd" \
     --transform-dir ${gmmdir}_ali \
     $dir data/$mic/train $gmmdir $dir/log $dir/data
  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10
fi

# Pre-train DBN, i.e. a stack of RBMs,
if [ $stage -le 1 ]; then
  dir=exp/$mic/dnn4_pretrain-dbn
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --rbm-iter 1 $data_fmllr/$mic/train $dir
fi

# Train the DNN optimizing per-frame cross-entropy,
if [ $stage -le 2 ]; then
  dir=exp/$mic/dnn4_pretrain-dbn_dnn
  ali=${gmmdir}_ali
  feature_transform=exp/$mic/dnn4_pretrain-dbn/final.feature_transform
  dbn=exp/$mic/dnn4_pretrain-dbn/6.dbn
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    $data_fmllr/$mic/train_tr90 $data_fmllr/$mic/train_cv10 data/lang $ali $ali $dir
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj $nj_decode --cmd "$decode_cmd" --config conf/decode_dnn.conf --acwt 0.1 \
    --num-threads 3 \
    $graph_dir $data_fmllr/$mic/dev $dir/decode_dev_${LM}
  steps/nnet/decode.sh --nj $nj_decode --cmd "$decode_cmd" --config conf/decode_dnn.conf --acwt 0.1 \
    --num-threads 3 \
    $graph_dir $data_fmllr/$mic/eval $dir/decode_eval_${LM}
fi


# Sequence training using sMBR criterion, we do Stochastic-GD with 
# per-utterance updates. We use usually good acwt 0.1.
# Lattices are not regenerated (it is faster).

dir=exp/$mic/dnn4_pretrain-dbn_dnn_smbr
srcdir=exp/$mic/dnn4_pretrain-dbn_dnn
acwt=0.1

# Generate lattices and alignments,
if [ $stage -le 3 ]; then
  steps/nnet/align.sh --nj $nj --cmd "$train_cmd" \
    $data_fmllr/$mic/train data/lang $srcdir ${srcdir}_ali
  steps/nnet/make_denlats.sh --nj $nj --cmd "$decode_cmd" --config conf/decode_dnn.conf \
    --acwt $acwt $data_fmllr/$mic/train data/lang $srcdir ${srcdir}_denlats
fi

# Re-train the DNN by 4 epochs of sMBR,
if [ $stage -le 4 ]; then
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 4 --acwt $acwt --do-smbr true \
    $data_fmllr/$mic/train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir
  # Decode (reuse HCLG graph)
  for ITER in 4 1; do
    steps/nnet/decode.sh --nj $nj_decode --cmd "$decode_cmd" --config conf/decode_dnn.conf \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $graph_dir $data_fmllr/$mic/dev $dir/decode_dev_${LM}_it${ITER}
    steps/nnet/decode.sh --nj $nj_decode --cmd "$decode_cmd" --config conf/decode_dnn.conf \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $graph_dir $data_fmllr/$mic/eval $dir/decode_eval_${LM}_it${ITER}
  done
fi

# Getting results [see RESULTS file]
# for x in exp/$mic/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done

