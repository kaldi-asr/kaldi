#!/bin/bash

# Copyright 2015 University of Sheffield (Jon Barker, Ricard Marxer)
#                Inria (Emmanuel Vincent)
#                Mitsubishi Electric Research Labs (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This script is made from the kaldi recipe of the 2nd CHiME Challenge Track 2
# made by Chao Weng

. ./path.sh
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

if [ $# -ne 1 ]; then
  printf "\nUSAGE: %s <CHiME3 root directory>\n\n" `basename $0`
  echo "Please specifies a CHiME3 root directory"
  echo "If you use kaldi scripts distributed in the CHiME3 data,"
  echo "It would be `pwd`/../.."
  exit 1;
fi

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

nj=30
# clean data
chime3_data=$1
wsj0_data=$chime3_data/data/WSJ0 # directory of WSJ0 in CHiME3. You can also specify your WSJ0 corpus directory

eval_flag=false # make it true when the evaluation data are released

# process for clean speech and making LMs etc. from original WSJ0
local/clean_wsj0_data_prep.sh $wsj0_data || exit 1;

local/wsj_prepare_dict.sh || exit 1;

utils/prepare_lang.sh data/local/dict "<SPOKEN_NOISE>" data/local/lang_tmp data/lang || exit 1;

local/clean_chime3_format_data.sh || exit 1;

# process for close talking speech for real data (will not be used)
local/real_close_chime3_data_prep.sh $chime3_data || exit 1;

# process for booth recording speech (will not be used)
local/bth_chime3_data_prep.sh $chime3_data || exit 1;

# process for distant talking speech for real and simulation data
local/real_noisy_chime3_data_prep.sh $chime3_data || exit 1;
local/simu_noisy_chime3_data_prep.sh $chime3_data || exit 1;

# Now make MFCC features for clean, close, and noisy data
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
# clean data
if $eval_flag; then
  list="tr05_orig_clean dt05_orig_clean et05_orig_clean"
else
  list="tr05_orig_clean dt05_orig_clean"
fi
# real data
if $eval_flag; then
  list=$list" tr05_real_close tr05_real_noisy dt05_real_close dt05_real_noisy et05_real_close et05_real_noisy"
else
  list=$list" tr05_real_close tr05_real_noisy dt05_real_close dt05_real_noisy"
fi
# simulation data
if $eval_flag; then
  list=$list" tr05_simu_noisy dt05_simu_noisy et05_simu_noisy"
else
  list=$list" tr05_simu_noisy dt05_simu_noisy"
fi
mfccdir=mfcc
for x in $list; do 
  steps/make_mfcc.sh --nj $nj \
    data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
done

# make mixed training set from real and simulation training data
# sr = simu + real
utils/combine_data.sh data/tr05_sr_noisy data/tr05_simu_noisy data/tr05_real_noisy
utils/combine_data.sh data/dt05_sr_noisy data/dt05_simu_noisy data/dt05_real_noisy

# training models for clean and noisy data
for train in tr05_sr_noisy tr05_real_noisy tr05_simu_noisy tr05_orig_clean; do
  nspk=`wc -l data/$train/spk2utt | awk '{print $1}'`
  if [ $nj -gt $nspk ]; then
    nj2=$nspk
  else
    nj2=$nj
  fi
  steps/train_mono.sh --boost-silence 1.25 --nj $nj2 \
    data/$train data/lang exp/mono0a_$train || exit 1;

  steps/align_si.sh --boost-silence 1.25 --nj $nj2 \
    data/$train data/lang exp/mono0a_$train exp/mono0a_ali_$train || exit 1;

  steps/train_deltas.sh --boost-silence 1.25 \
    2000 10000 data/$train data/lang exp/mono0a_ali_$train exp/tri1_$train || exit 1;

  steps/align_si.sh --nj $nj2 \
    data/$train data/lang exp/tri1_$train exp/tri1_ali_$train || exit 1;

  steps/train_lda_mllt.sh \
    --splice-opts "--left-context=3 --right-context=3" \
    2500 15000 data/$train data/lang exp/tri1_ali_$train exp/tri2b_$train || exit 1;

  steps/align_si.sh  --nj $nj2 \
    --use-graphs true data/$train data/lang exp/tri2b_$train exp/tri2b_ali_$train  || exit 1;

  steps/train_sat.sh \
    2500 15000 data/$train data/lang exp/tri2b_ali_$train exp/tri3b_$train || exit 1;

  utils/mkgraph.sh data/lang_test_tgpr_5k exp/tri3b_$train exp/tri3b_$train/graph_tgpr_5k || exit 1;

  # decode close speech
  steps/decode_fmllr.sh --nj 4 \
    exp/tri3b_$train/graph_tgpr_5k data/dt05_real_close exp/tri3b_$train/decode_tgpr_5k_dt05_real_close &
  # decode noisy speech
  steps/decode_fmllr.sh --nj 4 \
    exp/tri3b_$train/graph_tgpr_5k data/dt05_real_noisy exp/tri3b_$train/decode_tgpr_5k_dt05_real_noisy &
  # decode simu speech
  steps/decode_fmllr.sh --nj 4 \
    exp/tri3b_$train/graph_tgpr_5k data/dt05_simu_noisy exp/tri3b_$train/decode_tgpr_5k_dt05_simu_noisy &
done
wait

# get the best scores
for train in tr05_sr_noisy tr05_real_noisy tr05_simu_noisy tr05_orig_clean; do
  local/chime3_calc_wers.sh exp/tri3b_$train noisy \
      | tee exp/tri3b_$train/best_wer_noisy.result
done
