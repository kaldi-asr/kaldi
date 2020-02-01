#!/usr/bin/env bash

# Copyright 2015 University of Sheffield (Jon Barker, Ricard Marxer)
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

. utils/parse_options.sh || exit 1;

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

if [ $# -ne 2 ]; then
  printf "\nUSAGE: %s <enhancement method> <enhanced speech directory>\n\n" `basename $0`
  echo "First argument specifies a unique name for different enhancement method"
  echo "Second argument specifies the directory of enhanced wav files"
  exit 1;
fi

# set enhanced data
enhan=$1
enhan_data=$2

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# check whether run_init is executed
if [ ! -d data/lang ]; then
  echo "error, execute local/run_init.sh, first"
  exit 1;
fi

# process for enhanced data
if [ $stage -le 0 ]; then
  local/real_enhan_chime3_data_prep.sh $enhan $enhan_data
  local/simu_enhan_chime3_data_prep.sh $enhan $enhan_data
fi

# Now make MFCC features for clean, close, and noisy data
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc/$enhan
if [ $stage -le 1 ]; then
  for x in dt05_real_$enhan et05_real_$enhan tr05_real_$enhan dt05_simu_$enhan et05_simu_$enhan tr05_simu_$enhan; do
    steps/make_mfcc.sh --nj 8 --cmd "$train_cmd" \
      data/$x exp/make_mfcc/$x $mfccdir
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
  done
fi

# make mixed training set from real and simulation enhanced data
# multi = simu + real
if [ $stage -le 2 ]; then
  utils/combine_data.sh data/tr05_multi_$enhan data/tr05_simu_$enhan data/tr05_real_$enhan
  utils/combine_data.sh data/dt05_multi_$enhan data/dt05_simu_$enhan data/dt05_real_$enhan
  utils/combine_data.sh data/et05_multi_$enhan data/et05_simu_$enhan data/et05_real_$enhan
fi

# decode enhanced speech using clean AMs
if [ $stage -le 3 ]; then
  steps/decode_fmllr.sh --nj 4 --num-threads 3 --cmd "$decode_cmd" \
    exp/tri3b_tr05_orig_clean/graph_tgpr_5k data/dt05_real_$enhan exp/tri3b_tr05_orig_clean/decode_tgpr_5k_dt05_real_$enhan &
  steps/decode_fmllr.sh --nj 4 --num-threads 3 --cmd "$decode_cmd" \
    exp/tri3b_tr05_orig_clean/graph_tgpr_5k data/dt05_simu_$enhan exp/tri3b_tr05_orig_clean/decode_tgpr_5k_dt05_simu_$enhan &
  steps/decode_fmllr.sh --nj 4 --num-threads 3 --cmd "$decode_cmd" \
    exp/tri3b_tr05_orig_clean/graph_tgpr_5k data/et05_real_$enhan exp/tri3b_tr05_orig_clean/decode_tgpr_5k_et05_real_$enhan &
  steps/decode_fmllr.sh --nj 4 --num-threads 3 --cmd "$decode_cmd" \
    exp/tri3b_tr05_orig_clean/graph_tgpr_5k data/et05_simu_$enhan exp/tri3b_tr05_orig_clean/decode_tgpr_5k_et05_simu_$enhan &
fi

# training models using enhanced data
# training monophone model
if [ $stage -le 4 ]; then
  steps/train_mono.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
    data/tr05_multi_$enhan data/lang exp/mono0a_tr05_multi_$enhan

  steps/align_si.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
    data/tr05_multi_$enhan data/lang exp/mono0a_tr05_multi_$enhan exp/mono0a_ali_tr05_multi_$enhan
fi

# training triphone model with delta, delta+delta features
if [ $stage -le 5 ]; then
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/tr05_multi_$enhan data/lang exp/mono0a_ali_tr05_multi_$enhan exp/tri1_tr05_multi_$enhan
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/tr05_multi_$enhan data/lang exp/tri1_tr05_multi_$enhan exp/tri1_ali_tr05_multi_$enhan
fi

# training triphone model with lad mllt features
if [ $stage -le 6 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    2500 15000 data/tr05_multi_$enhan data/lang exp/tri1_ali_tr05_multi_$enhan exp/tri2b_tr05_multi_$enhan
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    --use-graphs true data/tr05_multi_$enhan data/lang exp/tri2b_tr05_multi_$enhan exp/tri2b_ali_tr05_multi_$enhan
fi

# training triphone model with SAT
if [ $stage -le 7 ]; then
  steps/train_sat.sh --cmd "$train_cmd" \
    2500 15000 data/tr05_multi_$enhan data/lang exp/tri2b_ali_tr05_multi_$enhan exp/tri3b_tr05_multi_$enhan
  utils/mkgraph.sh data/lang_test_tgpr_5k exp/tri3b_tr05_multi_$enhan exp/tri3b_tr05_multi_$enhan/graph_tgpr_5k
fi

# decode enhanced speech using AMs trained with enhanced data
if [ $stage -le 8 ]; then
  steps/decode_fmllr.sh --nj 4 --num-threads 3 --cmd "$decode_cmd" \
    exp/tri3b_tr05_multi_$enhan/graph_tgpr_5k data/dt05_real_$enhan exp/tri3b_tr05_multi_$enhan/decode_tgpr_5k_dt05_real_$enhan &
  steps/decode_fmllr.sh --nj 4 --num-threads 3 --cmd "$decode_cmd" \
    exp/tri3b_tr05_multi_$enhan/graph_tgpr_5k data/dt05_simu_$enhan exp/tri3b_tr05_multi_$enhan/decode_tgpr_5k_dt05_simu_$enhan &
  steps/decode_fmllr.sh --nj 4 --num-threads 3 --cmd "$decode_cmd" \
    exp/tri3b_tr05_multi_$enhan/graph_tgpr_5k data/et05_real_$enhan exp/tri3b_tr05_multi_$enhan/decode_tgpr_5k_et05_real_$enhan &
  steps/decode_fmllr.sh --nj 4 --num-threads 3 --cmd "$decode_cmd" \
    exp/tri3b_tr05_multi_$enhan/graph_tgpr_5k data/et05_simu_$enhan exp/tri3b_tr05_multi_$enhan/decode_tgpr_5k_et05_simu_$enhan &
  wait;
fi

# scoring
if [ $stage -le 9 ]; then
  # decoded results of enhanced speech using clean AMs
  local/chime3_calc_wers.sh exp/tri3b_tr05_orig_clean $enhan > exp/tri3b_tr05_orig_clean/best_wer_$enhan.result
  head -n 15 exp/tri3b_tr05_orig_clean/best_wer_$enhan.result
  # decoded results of enhanced speech using AMs trained with enhanced data
  local/chime3_calc_wers.sh exp/tri3b_tr05_multi_$enhan $enhan > exp/tri3b_tr05_multi_$enhan/best_wer_$enhan.result
  head -n 15 exp/tri3b_tr05_multi_$enhan/best_wer_$enhan.result
fi

echo "`basename $0` Done."
