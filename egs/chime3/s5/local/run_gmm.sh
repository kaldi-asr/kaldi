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

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

if [ $# -ne 2 ]; then
  printf "\nUSAGE: %s <enhancement method> <enhanced speech directory>\n\n" `basename $0`
  echo "First argument specifies a unique name for different enhancement method"
  echo "Second argument specifies the directory of enhanced wav files"
  exit 1;
fi

nj=30

# enhan data
enhan=$1
enhan_data=$2

# check whether run_init is executed
if [ ! -d data/lang ]; then
  echo "error, execute local/run_init.sh, first"
  exit 1;
fi

# process for enhan data
local/real_enhan_chime3_data_prep.sh $enhan $enhan_data || exit 1;
local/simu_enhan_chime3_data_prep.sh $enhan $enhan_data || exit 1;

# Now make MFCC features for clean, close, and noisy data
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc/$enhan
for x in dt05_real_$enhan tr05_real_$enhan dt05_simu_$enhan tr05_simu_$enhan; do 
  steps/make_mfcc.sh --nj $nj \
    data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
done

# make mixed training set from real and simulation enhancement training data
# multi = simu + real
utils/combine_data.sh data/tr05_multi_$enhan data/tr05_simu_$enhan data/tr05_real_$enhan
utils/combine_data.sh data/dt05_multi_$enhan data/dt05_simu_$enhan data/dt05_real_$enhan

# decode enhan speech using clean AMs
steps/decode_fmllr.sh --nj 4 --num-threads 4 \
  exp/tri3b_tr05_orig_clean/graph_tgpr_5k data/dt05_real_$enhan exp/tri3b_tr05_orig_clean/decode_tgpr_5k_dt05_real_$enhan &
steps/decode_fmllr.sh --nj 4 --num-threads 4 \
  exp/tri3b_tr05_orig_clean/graph_tgpr_5k data/dt05_simu_$enhan exp/tri3b_tr05_orig_clean/decode_tgpr_5k_dt05_simu_$enhan &

# training models using enhan data
steps/train_mono.sh --boost-silence 1.25 --nj $nj \
  data/tr05_multi_$enhan data/lang exp/mono0a_tr05_multi_$enhan || exit 1;

steps/align_si.sh --boost-silence 1.25 --nj $nj \
  data/tr05_multi_$enhan data/lang exp/mono0a_tr05_multi_$enhan exp/mono0a_ali_tr05_multi_$enhan || exit 1;

steps/train_deltas.sh --boost-silence 1.25 \
  2000 10000 data/tr05_multi_$enhan data/lang exp/mono0a_ali_tr05_multi_$enhan exp/tri1_tr05_multi_$enhan || exit 1;

steps/align_si.sh --nj $nj \
  data/tr05_multi_$enhan data/lang exp/tri1_tr05_multi_$enhan exp/tri1_ali_tr05_multi_$enhan || exit 1;

steps/train_lda_mllt.sh \
  --splice-opts "--left-context=3 --right-context=3" \
  2500 15000 data/tr05_multi_$enhan data/lang exp/tri1_ali_tr05_multi_$enhan exp/tri2b_tr05_multi_$enhan || exit 1;

steps/align_si.sh  --nj $nj \
  --use-graphs true data/tr05_multi_$enhan data/lang exp/tri2b_tr05_multi_$enhan exp/tri2b_ali_tr05_multi_$enhan  || exit 1;

steps/train_sat.sh \
  2500 15000 data/tr05_multi_$enhan data/lang exp/tri2b_ali_tr05_multi_$enhan exp/tri3b_tr05_multi_$enhan || exit 1;

utils/mkgraph.sh data/lang_test_tgpr_5k exp/tri3b_tr05_multi_$enhan exp/tri3b_tr05_multi_$enhan/graph_tgpr_5k || exit 1;

# decode enhan speech using enhan AMs
steps/decode_fmllr.sh --nj 4 --num-threads 4 \
  exp/tri3b_tr05_multi_$enhan/graph_tgpr_5k data/dt05_real_$enhan exp/tri3b_tr05_multi_$enhan/decode_tgpr_5k_dt05_real_$enhan &
steps/decode_fmllr.sh --nj 4 --num-threads 4 \
  exp/tri3b_tr05_multi_$enhan/graph_tgpr_5k data/dt05_simu_$enhan exp/tri3b_tr05_multi_$enhan/decode_tgpr_5k_dt05_simu_$enhan &

wait;
# decoded results of enhan speech using clean AMs
local/chime3_calc_wers.sh exp/tri3b_tr05_orig_clean $enhan > exp/tri3b_tr05_orig_clean/best_wer_$enhan.result
head -n 11 exp/tri3b_tr05_orig_clean/best_wer_$enhan.result
# decoded results of enhan speech using enhan AMs
local/chime3_calc_wers.sh exp/tri3b_tr05_multi_$enhan $enhan > exp/tri3b_tr05_multi_$enhan/best_wer_$enhan.result
head -n 11 exp/tri3b_tr05_multi_$enhan/best_wer_$enhan.result
