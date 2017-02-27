#!/bin/bash -v

# Apache 2.0,
# Copyright: 2017 University of Sheffield (author: Yulan Liu) 
#            2017 Brno University of Technology (author: Karel Vesely)

. ./cmd.sh
. ./path.sh

# DNN training. This script is based on egs/swbd/s5b/local/run_dnn.sh

# Config:
stage=0 # resume training with --stage=N
. utils/parse_options.sh || exit 1;
#

if [ $# -ne 2 ]; then
  printf "\nUSAGE: %s [opts] <mic condition(ihm|sdm|mdm)> <dataset mode>\n\n" `basename $0`
  exit 1;
fi
mic=$1
MODE=$2

gmmdir=exp/$mic/$MODE/tri4a
data_fmllr=data_${mic}-fmllr-tri4

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7
graph_dir=$gmmdir/graph_${LM}

# Set bash to 'debug' mode, it will exit on : 
# -e 'error', -u 'undefined variable', -x 'print commands', -o ... 'error in pipeline',
set -euxo pipefail

nj_train=$(cat data/$mic/$MODE/train/spk2utt | wc -l)
nj_dev=$(cat data/$mic/$MODE/dev/spk2utt | wc -l)
nj_eval=$(cat data/$mic/$MODE/eval/spk2utt | wc -l)

# Store fMLLR features, so we can train on them easily,
if [ $stage -le 0 ]; then
  # eval
  dir=$data_fmllr/$mic/$MODE/eval
  steps/nnet/make_fmllr_feats.sh --nj $nj_eval --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode_eval_${LM} \
     $dir data/$mic/$MODE/eval $gmmdir $dir/log $dir/data
  # dev
  dir=$data_fmllr/$mic/$MODE/dev
  steps/nnet/make_fmllr_feats.sh --nj $nj_dev --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode_dev_${LM} \
     $dir data/$mic/$MODE/dev $gmmdir $dir/log $dir/data
  # train
  dir=$data_fmllr/$mic/$MODE/train
  steps/nnet/make_fmllr_feats.sh --nj $nj_train --cmd "$train_cmd" \
     --transform-dir ${gmmdir}_ali \
     $dir data/$mic/$MODE/train $gmmdir $dir/log $dir/data
  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10
fi

# Pre-train DBN, i.e. a stack of RBMs,
if [ $stage -le 1 ]; then
  dir=exp/$mic/$MODE/dnn4_pretrain-dbn
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --rbm-iter 1 $data_fmllr/$mic/$MODE/train $dir
fi

# Train the DNN optimizing per-frame cross-entropy,
if [ $stage -le 2 ]; then
  dir=exp/$mic/$MODE/dnn4_pretrain-dbn_dnn
  ali=${gmmdir}_ali
  feature_transform=exp/$mic/$MODE/dnn4_pretrain-dbn/final.feature_transform
  dbn=exp/$mic/$MODE/dnn4_pretrain-dbn/6.dbn
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    $data_fmllr/$mic/$MODE/train_tr90 $data_fmllr/$mic/$MODE/train_cv10 data/lang $ali $ali $dir
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj $nj_dev --cmd "$decode_large_cmd" --config conf/decode_dnn.conf --acwt 0.1 \
    --num-threads 3 \
    $graph_dir $data_fmllr/$mic/$MODE/dev $dir/decode_dev_${LM}
  steps/nnet/decode.sh --nj $nj_eval --cmd "$decode_large_cmd" --config conf/decode_dnn.conf --acwt 0.1 \
    --num-threads 3 \
    $graph_dir $data_fmllr/$mic/$MODE/eval $dir/decode_eval_${LM}
fi

### exit 0 # We can end here,

# Sequence training using sMBR criterion, we do Stochastic-GD with 
# per-utterance updates. We use usually good acwt 0.1.
# Lattices are not regenerated (it is faster).

dir=exp/$mic/$MODE/dnn4_pretrain-dbn_dnn_smbr
srcdir=exp/$mic/$MODE/dnn4_pretrain-dbn_dnn
acwt=0.1

# Generate lattices and alignments,
if [ $stage -le 3 ]; then
  steps/nnet/align.sh --nj $nj_train --cmd "$train_cmd" \
    $data_fmllr/$mic/$MODE/train data/lang $srcdir ${srcdir}_ali
  steps/nnet/make_denlats.sh --nj $nj_train --cmd "$decode_large_cmd" --config conf/decode_dnn.conf \
    --acwt $acwt $data_fmllr/$mic/$MODE/train data/lang $srcdir ${srcdir}_denlats
fi

# Re-train the DNN by 4 epochs of sMBR,
if [ $stage -le 4 ]; then
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 4 --acwt $acwt --do-smbr true \
    $data_fmllr/$mic/$MODE/train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir
  # Decode (reuse HCLG graph)
  for ITER in 4 1; do
    steps/nnet/decode.sh --nj $nj_dev --cmd "$decode_large_cmd" --config conf/decode_dnn.conf \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $graph_dir $data_fmllr/$mic/$MODE/dev $dir/decode_dev_${LM}_it${ITER}
    steps/nnet/decode.sh --nj $nj_eval --cmd "$decode_large_cmd" --config conf/decode_dnn.conf \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $graph_dir $data_fmllr/$mic/$MODE/eval $dir/decode_eval_${LM}_it${ITER}
  done
fi

# Getting results [see RESULTS file]
# for x in exp/$mic/$MODE/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
