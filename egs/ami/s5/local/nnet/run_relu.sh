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

set -euxo pipefail

# Store fMLLR features, so we can train on them easily,
if [ $stage -le 0 -a ! -e $data_fmllr/$mic/eval ]; then
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

train=data_ihm-fmllr-tri4/ihm/train
dev=data_ihm-fmllr-tri4/ihm/dev
eval=data_ihm-fmllr-tri4/ihm/eval

lrate=0.00025
param_std=0.02
lr_alpha=1.0
lr_beta=0.75
dropout_schedule=0.2,0.2,0.2,0.2,0.2,0.0
gmm=$gmmdir
graph=$graph_dir

# Train 6 layer DNN from random initialization,
# - Parametric RELU, alphas+betas trained,
# - Dropout retention 0.8 in 5 initial epochs with fixed learning rate,
if [ $stage -le 1 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/$mic/dnn4d-6L1024-relu
  ali=${gmm}_ali
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --learn-rate $lrate \
    --splice 5 --hid-layers 6 --hid-dim 1024 \
    --proto-opts "--activation-type=<ParametricRelu> --activation-opts=<AlphaLearnRateCoef>_${lr_alpha}_<BetaLearnRateCoef>_${lr_beta} --param-stddev-factor $param_std --hid-bias-mean 0 --hid-bias-range 0 --with-dropout --no-glorot-scaled-stddev --no-smaller-input-weights" \
    --scheduler-opts "--keep-lr-iters 5 --dropout-schedule $dropout_schedule" \
    ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --acwt 0.1 \
    $graph $dev $dir/decode_$(basename $dev)
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --acwt 0.1 \
    $graph $eval $dir/decode_$(basename $eval)
fi

# Sequence training using sMBR criterion, we do Stochastic-GD with
# per-utterance updates. We use usually good acwt 0.1.
# Lattices are not regenerated (it is faster).

dir=exp/$mic/dnn4d-6L1024-relu_smbr
srcdir=exp/$mic/dnn4d-6L1024-relu
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
    --learn-rate 0.0000003 \
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

