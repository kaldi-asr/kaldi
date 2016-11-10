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

if [ $# -ne 3 ]; then
  printf "\nUSAGE: %s <enhancement method> <enhanced speech directory> <model dir>\n\n" `basename $0`
  echo "First argument specifies a unique name for different enhancement method"
  echo "Second argument specifies the directory of enhanced wav files"
  echo "Third argument specifies acoustic and language model directory"
  exit 1;
fi

# set enhanced data
enhan=$1
enhan_data=$2
# set model directory
mdir=$3

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

# process for enhanced data
if [ $stage -le 0 ]; then
  if [ ! -d data/dt05_real_$enhan ] || [ ! -d data/et05_real_$enhan ]; then
    local/real_enhan_chime4_data_prep.sh $enhan $enhan_data
    local/simu_enhan_chime4_data_prep.sh $enhan $enhan_data
  fi
fi

# Now make MFCC features for enhanced data
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc/$enhan
if [ $stage -le 1 ]; then
  if $eval_flag; then
    tasks="dt05_real_$enhan dt05_simu_$enhan et05_real_$enhan et05_simu_$enhan"
  else
    tasks="dt05_real_$enhan dt05_simu_$enhan"
  fi
  for x in $tasks; do
    if [ ! -e data/$x/feats.scp ]; then
      steps/make_mfcc.sh --nj 8 --cmd "$train_cmd" \
	data/$x exp/make_mfcc/$x $mfccdir
      steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
    fi
  done
fi

# make mixed training set from real and simulation enhanced data
# multi = simu + real
if [ $stage -le 2 ]; then
  if [ ! -d data/dt05_multi_$enhan ] || [ ! -d data/et05_multi_$enhan ]; then
    utils/combine_data.sh data/dt05_multi_$enhan data/dt05_simu_$enhan data/dt05_real_$enhan
    if $eval_flag; then
    utils/combine_data.sh data/et05_multi_$enhan data/et05_simu_$enhan data/et05_real_$enhan
    fi
  fi
fi

# decode enhanced speech using AMs trained with enhanced data
if [ $stage -le 3 ]; then
  steps/decode_fmllr.sh --nj 4 --num-threads 3 --cmd "$decode_cmd" \
    exp/tri3b_tr05_multi_${train}/graph_tgpr_5k data/dt05_real_$enhan exp/tri3b_tr05_multi_${train}/decode_tgpr_5k_dt05_real_$enhan &
  steps/decode_fmllr.sh --nj 4 --num-threads 3 --cmd "$decode_cmd" \
    exp/tri3b_tr05_multi_${train}/graph_tgpr_5k data/dt05_simu_$enhan exp/tri3b_tr05_multi_${train}/decode_tgpr_5k_dt05_simu_$enhan &
  if $eval_flag; then
  steps/decode_fmllr.sh --nj 4 --num-threads 3 --cmd "$decode_cmd" \
    exp/tri3b_tr05_multi_${train}/graph_tgpr_5k data/et05_real_$enhan exp/tri3b_tr05_multi_${train}/decode_tgpr_5k_et05_real_$enhan &
  steps/decode_fmllr.sh --nj 4 --num-threads 3 --cmd "$decode_cmd" \
    exp/tri3b_tr05_multi_${train}/graph_tgpr_5k data/et05_simu_$enhan exp/tri3b_tr05_multi_${train}/decode_tgpr_5k_et05_simu_$enhan &
  fi
  wait;
fi

# scoring
if [ $stage -le 4 ]; then
  # decoded results of enhanced speech using AMs trained with enhanced data
  local/chime4_calc_wers.sh exp/tri3b_tr05_multi_${train} $enhan exp/tri3b_tr05_multi_${train}/graph_tgpr_5k \
    > exp/tri3b_tr05_multi_${train}/best_wer_$enhan.result
  head -n 15 exp/tri3b_tr05_multi_${train}/best_wer_$enhan.result
fi

echo "`basename $0` Done."
