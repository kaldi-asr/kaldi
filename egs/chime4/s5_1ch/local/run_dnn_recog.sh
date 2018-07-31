#!/bin/bash

# Copyright 2016 University of Sheffield (Jon Barker, Ricard Marxer)
#                Inria (Emmanuel Vincent)
#                Mitsubishi Electric Research Labs (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This script is made from the kaldi recipe of the 2nd CHiME Challenge Track 2
# made by Chao Weng

. ./path.sh
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

# Config:
nj=30
stage=0 # resume training with --stage=N
train=noisy
eval_flag=true # make it true when the evaluation data are released

. utils/parse_options.sh || exit 1;

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

if [ $# -ne 2 ]; then
  printf "\nUSAGE: %s <enhancement method> <model dir>\n\n" `basename $0`
  echo "First argument specifies a unique name for different enhancement method"
  echo "Second argument specifies acoustic and language model directory"
  exit 1;
fi

# set enhanced data
enhan=$1
# set model directory
mdir=$2

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# check data/loca/data
if [ ! -d $mdir/data/local/data ]; then
  echo "error, set $mdir correctly"
  exit 1;
elif [ ! -d data/local/data ]; then
  echo "copy $mdir/data/local/data"
  mkdir -p data/local
  cp -r $mdir/data/local/data data/local/
fi

# check gmm model
if [ ! -d $mdir/exp/tri3b_tr05_multi_${train} ]; then
  echo "error, set $mdir correctly"
  exit 1;
elif [ ! -d exp/tri3b_tr05_multi_${train} ]; then
  echo "copy $mdir/exp/tri3b_tr05_multi_${train}"
  mkdir -p exp
  cp -r $mdir/exp/tri3b_tr05_multi_${train} exp/
fi

# check dnn graph
if [ ! -d $mdir/exp/tri4a_dnn_tr05_multi_${train}/graph_tgpr_5k ]; then
  echo "error, set $mdir correctly"
  exit 1;
elif [ ! -d exp/tri4a_dnn_tr05_multi_${train}/graph_tgpr_5k ]; then
  echo "copy $mdir/exp/tri4a_dnn_tr05_multi_${train}/graph_tgpr_5k"
  mkdir -p exp/tri4a_dnn_tr05_multi_${train}
  cp -r $mdir/exp/tri4a_dnn_tr05_multi_${train}/graph_tgpr_5k exp/tri4a_dnn_tr05_multi_${train}/
fi

# check dnn smbr model
if [ ! -d $mdir/exp/tri4a_dnn_tr05_multi_${train}_smbr_i1lats ]; then
  echo "error, set $mdir correctly"
  exit 1;
elif [ ! -d exp/tri4a_dnn_tr05_multi_${train}_smbr_i1lats ]; then
  echo "copy $mdir/exp/tri4a_dnn_tr05_multi_${train}_smbr_i1lats"
  mkdir -p exp
  cp -r $mdir/exp/tri4a_dnn_tr05_multi_${train}_smbr_i1lats exp/
fi

# make fmllr feature for dev and eval
gmmdir=exp/tri3b_tr05_multi_${train}
data_fmllr=data-fmllr-tri3b
mkdir -p $data_fmllr
fmllrdir=fmllr-tri3b/$enhan
if [ $stage -le 4 ]; then
  if $eval_flag; then
    tasks="dt05_real_$enhan dt05_simu_$enhan et05_real_$enhan et05_simu_$enhan"
  else
    tasks="dt05_real_$enhan dt05_simu_$enhan"
  fi
  for x in $tasks; do
    steps/nnet/make_fmllr_feats.sh --nj 4 --cmd "$train_cmd" \
      --transform-dir $gmmdir/decode_tgpr_5k_$x \
      $data_fmllr/$x data/$x $gmmdir exp/make_fmllr_tri3b/$x $fmllrdir
  done
fi

# make mixed training set from real and simulation enhanced data
# multi = simu + real
if [ $stage -le 5 ]; then
  utils/combine_data.sh $data_fmllr/dt05_multi_$enhan $data_fmllr/dt05_simu_$enhan $data_fmllr/dt05_real_$enhan
  if $eval_flag; then
  utils/combine_data.sh $data_fmllr/et05_multi_$enhan $data_fmllr/et05_simu_$enhan $data_fmllr/et05_real_$enhan
  fi
fi

# Re-generate lattices, run 4 more sMBR iterations
dir=exp/tri4a_dnn_tr05_multi_${train}_smbr_i1lats
acwt=0.1

# Decode (reuse HCLG graph)
if [ $stage -le 6 ]; then
  for ITER in 1 2 3 4; do
    steps/nnet/decode.sh --nj 4 --num-threads 3 --cmd "$decode_cmd" --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      exp/tri4a_dnn_tr05_multi_${train}/graph_tgpr_5k $data_fmllr/dt05_real_${enhan} $dir/decode_tgpr_5k_dt05_real_${enhan}_it${ITER} &
    steps/nnet/decode.sh --nj 4 --num-threads 3 --cmd "$decode_cmd" --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      exp/tri4a_dnn_tr05_multi_${train}/graph_tgpr_5k $data_fmllr/dt05_simu_${enhan} $dir/decode_tgpr_5k_dt05_simu_${enhan}_it${ITER} &
    if $eval_flag; then
    steps/nnet/decode.sh --nj 4 --num-threads 3 --cmd "$decode_cmd" --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      exp/tri4a_dnn_tr05_multi_${train}/graph_tgpr_5k $data_fmllr/et05_real_${enhan} $dir/decode_tgpr_5k_et05_real_${enhan}_it${ITER} &
    steps/nnet/decode.sh --nj 4 --num-threads 3 --cmd "$decode_cmd" --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      exp/tri4a_dnn_tr05_multi_${train}/graph_tgpr_5k $data_fmllr/et05_simu_${enhan} $dir/decode_tgpr_5k_et05_simu_${enhan}_it${ITER} &
    fi
    wait
  done
fi

# scoring
if [ $stage -le 7 ]; then
  # decoded results of enhanced speech using sequence-training DNN
  ./local/chime4_calc_wers_smbr.sh $dir ${enhan} exp/tri4a_dnn_tr05_multi_${train}/graph_tgpr_5k > $dir/best_wer_${enhan}.result
  head -n 15 $dir/best_wer_${enhan}.result
fi

echo "`basename $0` Done."
