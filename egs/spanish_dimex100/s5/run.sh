#!/usr/bin/env bash

. ./path.sh || exit 1
. ./cmd.sh || exit 1

########
# Config
########

train_cmd="utils/run.pl"
decode_cmd="utils/run.pl"

CORPUS_DIR="CorpusDimex100"

N_HMM=2000 # leaves
N_GAUSSIANS=11000


#################
# Download corpus
#################

echo
echo Downloading corpus
echo
if [ ! -d "$CORPUS_DIR" ]; then
  wget http://turing.iimas.unam.mx/~luis/DIME/DIMEx100/DVD/DVDCorpusDimex100.zip || exit 1;
  unzip DVDCorpusDimex100.zip || exit 1;
fi


##################
# Data preparation
##################

echo
echo Data preparation
echo
rm -rf data exp mfcc
local/data_prep.sh "$CORPUS_DIR"
utils/fix_data_dir.sh "data/train"
utils/fix_data_dir.sh "data/test"


#####################
# Features generation
#####################

echo
echo Features generation
echo
steps/make_mfcc.sh --cmd "$train_cmd" "data/train" "exp/make_mfcc/train" mfcc
steps/make_mfcc.sh --cmd "$train_cmd" "data/test"  "exp/make_mfcc/test"  mfcc

steps/compute_cmvn_stats.sh "data/train" "exp/make_mfcc/train" mfcc
steps/compute_cmvn_stats.sh "data/test" "exp/make_mfcc/test" mfcc

utils/validate_data_dir.sh "data/train"
utils/validate_data_dir.sh "data/test"


#######################
# Lang data preparation
#######################

echo
echo Language data preparation
echo
rm -rf data/local/dict
local/lang_prep.sh "$CORPUS_DIR"
utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang
utils/fix_data_dir.sh "data/train"
utils/fix_data_dir.sh "data/test"


############################
# Language model preparation
############################

echo
echo Language model preparation
echo
local/lm_prep.sh


#######################
# Training and Decoding
#######################

echo
echo Training
echo
# utils/subset_data_dir.sh --first data/train 500 data/train_500

# Training and aligning
steps/train_mono.sh --cmd "$train_cmd" data/train data/lang exp/mono || exit 1
steps/align_si.sh --cmd "$train_cmd" data/train data/lang exp/mono exp/mono_aligned || exit 1
steps/train_deltas.sh "$N_HMM" "$N_GAUSSIANS" data/train data/lang exp/mono_aligned exp/tri1 || exit 1
steps/align_si.sh --cmd "$train_cmd" data/train data/lang exp/tri1 exp/tri1_aligned || exit 1

# train tri2b [LDA+MLLT]
steps/train_lda_mllt.sh --cmd "$train_cmd" "$N_HMM" "$N_GAUSSIANS" data/train data/lang exp/tri1_aligned exp/tri2b || exit 1;
utils/mkgraph.sh data/lang exp/tri2b exp/tri2b/graph
steps/align_si.sh --cmd "$train_cmd" data/train data/lang exp/tri2b exp/tri2b_aligned || exit 1

#  Do MMI on top of LDA+MLLT.
steps/make_denlats.sh --cmd "$train_cmd" data/train data/lang exp/tri2b exp/tri2b_denlats || exit 1;
steps/train_mmi.sh --boost 0.05 data/train data/lang exp/tri2b_aligned exp/tri2b_denlats exp/tri2b_mmi_b0.05 || exit 1;



# Decoding
echo
echo Decoding
echo
steps/decode.sh --config conf/decode.config --cmd "$decode_cmd" exp/tri2b/graph data/test exp/tri2b_mmi_b0.05/decode_test

for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
