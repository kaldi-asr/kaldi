#!/bin/bash

#
# Copyright 2018 Yuan-Fu Liao @ National Taipei University of Technology
#
# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.
# Caution: some of the graph creation steps use quite a bit of memory, so you
# should run this on a machine that has sufficient memory.

#
# shell options
#

set -e -o pipefail

#
# jobs configuration
#

. ./cmd.sh

numjobs=100
numjobs_test=100

#
# prepare speeech and LM data
#

#
# Lexicon Preparation,
#

echo
echo "========== prepare_dict: Lexicon Preparation =========="
echo

true && (

local/prepare_dict.sh || exit 1;

)

echo
echo "========== prepare_dict done =========="
echo

#
# Data Preparation,
#

echo
echo "========== prepare_data: Data Preparation =========="
echo

true && (

local/prepare_data.sh || exit 1;

)

echo
echo "========== prepare_data done =========="
echo

#
# Phone Sets, questions, L compilation
#

echo
echo "========== prepare_lang: Phone Sets, questions, L compilation =========="
echo

true && (

rm -rf data/lang
utils/prepare_lang.sh --position-dependent-phones false data/local/dict \
    "<SIL>" data/local/lang data/lang || exit 1;

)

echo
echo "========== prepare_lang done =========="
echo

#
# LM training
#

echo
echo "========== train_lms: LM training =========="
echo

true && (

rm -rf data/local/lm/3gram-mincount
local/train_lms.sh || exit 1;

)

echo
echo "========== train_lms done =========="
echo

#
# G compilation, check LG composition
#

echo
echo "========== format_lm: G compilation, check LG composition =========="
echo

true && (
utils/format_lm.sh data/lang data/local/lm/3gram-mincount/lm_unpruned.gz \
    data/local/dict/lexicon.txt data/lang_test || exit 1;

)

echo
echo "========== format_lm done =========="
echo

# Now make MFCC plus pitch features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.

mfccdir=mfcc

#
# mfcc
#

echo
echo "========== make_mfcc_pitch: mfcc =========="
echo

true && (

for x in train test; do
  steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj $numjobs data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  utils/fix_data_dir.sh data/$x || exit 1;
done

)

echo
echo "========== make_mfcc_pitch done =========="
echo

#
#
#

# Make some small data subsets for early system-build stages.  Note, there are 29k
# utterances in the train_clean_100 directory which has 100 hours of data.
# For the monophone stages we select the shortest utterances, which should make it
# easier to align the data from a flat start.

echo
echo "========== make training subsets =========="
echo

true && (
utils/subset_data_dir.sh --shortest	data/train   3000  data/train_mono
utils/subset_data_dir.sh		data/train   6000  data/train_tri1
utils/subset_data_dir.sh		data/train   9000  data/train_tri2
utils/subset_data_dir.sh		data/train  12000  data/train_tri3
utils/subset_data_dir.sh		data/train  15000  data/train_tri4
)

echo
echo "========== make training subsets done =========="
echo

#
# mono
#

echo
echo "========== train_mono: mono done =========="
echo

true && (

steps/train_mono.sh --boost-silence 1.25 --cmd "$train_cmd" --nj $numjobs \
  data/train_mono data/lang exp/mono || exit 1;

# Get alignments from monophone system.
steps/align_si.sh --boost-silence 1.25 --cmd "$train_cmd" --nj $numjobs \
  data/train_tri1 data/lang exp/mono exp/mono_ali_tri1 || exit 1;

)

# Monophone decoding
true &&
(
utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph || exit 1;
steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj $numjobs_test \
  exp/mono/graph data/test exp/mono/decode_test
)&


echo
echo "========== train_mono done =========="
echo

#
# tri1
#

echo
echo "========== tri1 =========="
echo

true && (

# train tri1 [first triphone pass]
steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
 2500 20000 data/train_tri1 data/lang exp/mono_ali_tri1 exp/tri1 || exit 1;

# align tri1
steps/align_si.sh --cmd "$train_cmd" --nj $numjobs \
  data/train_tri2 data/lang exp/tri1 exp/tri1_ali_tri2 || exit 1;

)

# decode tri1
true && (
utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph || exit 1;
steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj $numjobs_test \
  exp/tri1/graph data/test exp/tri1/decode_test
)&

echo
echo "========== tri1 done =========="
echo

#
# tri2
#

echo
echo "========== tri2 =========="
echo

true && (

# train tri2 [delta+delta-deltas]
steps/train_deltas.sh --cmd "$train_cmd" \
 2500 20000 data/train_tri2 data/lang exp/tri1_ali_tri2 exp/tri2 || exit 1;

# align tri2b
steps/align_si.sh --cmd "$train_cmd" --nj $numjobs \
  data/train_tri3 data/lang exp/tri2 exp/tri2_ali_tri3 || exit 1;

)

# decode tri2
true && (
utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph
steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj $numjobs_test \
  exp/tri2/graph data/test exp/tri2/decode_test
)&


echo
echo "========== tri2 done =========="
echo

#
# tri3a
#

echo
echo "========== tri3 =========="
echo

true && (

# Train tri3a, which is LDA+MLLT,
steps/train_lda_mllt.sh --cmd "$train_cmd" \
 2500 20000 data/train_tri3 data/lang exp/tri2_ali_tri3 exp/tri3a || exit 1;

)

# decode tri3a
true && (
utils/mkgraph.sh data/lang_test exp/tri3a exp/tri3a/graph || exit 1;
steps/decode.sh --cmd "$decode_cmd" --nj $numjobs_test --config conf/decode.config \
  exp/tri3a/graph data/test exp/tri3a/decode_test
)&

echo
echo "========== tri3 done =========="
echo

#
# tri4
#

echo
echo "========== tri4 =========="
echo

true && (

# From now, we start building a more serious system (with SAT), and we'll
# do the alignment with fMLLR.
steps/align_fmllr.sh --cmd "$train_cmd" --nj $numjobs \
  data/train_tri4 data/lang exp/tri3a exp/tri3a_ali_tri4 || exit 1;

steps/train_sat.sh --cmd "$train_cmd" \
  2500 20000 data/train_tri4 data/lang exp/tri3a_ali_tri4 exp/tri4a || exit 1;

# align tri4a
steps/align_fmllr.sh  --cmd "$train_cmd" --nj $numjobs \
  data/train data/lang exp/tri4a exp/tri4a_ali

)

# decode tri4a
true && (
utils/mkgraph.sh data/lang_test exp/tri4a exp/tri4a/graph
steps/decode_fmllr.sh --cmd "$decode_cmd" --nj $numjobs_test --config conf/decode.config \
  exp/tri4a/graph data/test exp/tri4a/decode_test
)&

echo
echo "========== tri4 done =========="
echo

#
# tri5
#

echo
echo "========== tri5 =========="
echo

true && (

# Building a larger SAT system.
steps/train_sat.sh --cmd "$train_cmd" \
  3500 100000 data/train data/lang exp/tri4a_ali exp/tri5a || exit 1;

# align tri5a
steps/align_fmllr.sh --cmd "$train_cmd" --nj $numjobs \
  data/train data/lang exp/tri5a exp/tri5a_ali || exit 1;

)

# decode tri5
true && (
utils/mkgraph.sh data/lang_test exp/tri5a exp/tri5a/graph || exit 1;
steps/decode_fmllr.sh --cmd "$decode_cmd" --nj $numjobs_test --config conf/decode.config \
   exp/tri5a/graph data/test exp/tri5a/decode_test || exit 1;
)&

echo
echo "========== tri5 done =========="
echo

#
# nnet3 tdnn models
#

echo
echo "========== nnet3 =========="
echo

true && (

# option to bypass certain steps, for example
# local/nnet3/run_tdnn.sh --stage 8 --train_stage 0
# local/nnet3/run_tdnn.sh --stage 9
#
local/nnet3/run_tdnn.sh

)

echo
echo "========== nnet3 done =========="
echo

#
# chain model
#

echo
echo "========== chain =========="
echo

true && (

# There are options to bypass certain steps, for example
# local/chain/run_tdnn.sh --stage 11 --train_stage 0
local/chain/run_tdnn.sh

)

echo
echo "========== chain done =========="
echo

#
# nnet3 getting results (see RESULTS file)
#

echo
echo "========== result =========="
echo

for x in exp/*/decode_test; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
for x in exp/*/*/decode_test; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null

echo
echo "========== result done =========="
echo

exit 0;
