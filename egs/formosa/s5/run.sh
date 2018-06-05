#!/bin/bash
#
# Copyright 2018, Yuan-Fu Liao, National Taipei University of Technology, yfliao@mail.ntut.edu.tw
#
# Instruction
#
# Before you run this recips, please apply, download and put or make a link of the corpus under this folder (folder name: "NER-Trs-Vol1").
#
# Then, you could excute "run.sh" to train models and decode test data. There are many switchs in this sceript (flag "true" or "false"). you could set them to "false" to bypass centain steps. ), for example:
#
# false && (
#    local/prepare_dict.sh || exit 1;
# )
#
# Finally, run "result.sh" to collect all decoding results.
#
# Reference results
#
# WER:
#
# %WER 61.32 [ 83373 / 135972, 5458 ins, 19156 del, 58759 sub ] exp/mono/decode_test/wer_11_0.0
# %WER 41.00 [ 55742 / 135972, 6725 ins, 12763 del, 36254 sub ] exp/tri1/decode_test/wer_15_0.0
# %WER 40.41 [ 54948 / 135972, 7366 ins, 11505 del, 36077 sub ] exp/tri2/decode_test/wer_14_0.0
# %WER 38.67 [ 52574 / 135972, 6855 ins, 11250 del, 34469 sub ] exp/tri3a/decode_test/wer_15_0.0
# %WER 35.70 [ 48546 / 135972, 7197 ins, 9717 del, 31632 sub ] exp/tri4a/decode_test/wer_17_0.0
# %WER 39.70 [ 53982 / 135972, 7199 ins, 11014 del, 35769 sub ] exp/tri4a/decode_test.si/wer_15_0.0
# %WER 32.11 [ 43661 / 135972, 6112 ins, 10185 del, 27364 sub ] exp/tri5a/decode_test/wer_17_0.5
# %WER 35.93 [ 48849 / 135972, 6611 ins, 10427 del, 31811 sub ] exp/tri5a/decode_test.si/wer_13_0.5
# %WER 24.43 [ 33218 / 135972, 5524 ins, 7583 del, 20111 sub ] exp/nnet3/tdnn_sp/decode_test/wer_12_0.0
#
# CER:
#
# %WER 54.09 [ 116688 / 215718, 4747 ins, 24510 del, 87431 sub ] exp/mono/decode_test/cer_10_0.0
# %WER 32.61 [ 70336 / 215718, 5866 ins, 16282 del, 48188 sub ] exp/tri1/decode_test/cer_13_0.0
# %WER 32.10 [ 69238 / 215718, 6186 ins, 15772 del, 47280 sub ] exp/tri2/decode_test/cer_13_0.0
# %WER 30.40 [ 65583 / 215718, 6729 ins, 13115 del, 45739 sub ] exp/tri3a/decode_test/cer_12_0.0
# %WER 27.53 [ 59389 / 215718, 6311 ins, 13008 del, 40070 sub ] exp/tri4a/decode_test/cer_15_0.0
# %WER 31.42 [ 67779 / 215718, 6565 ins, 13660 del, 47554 sub ] exp/tri4a/decode_test.si/cer_12_0.0
# %WER 24.21 [ 52232 / 215718, 6425 ins, 11543 del, 34264 sub ] exp/tri5a/decode_test/cer_15_0.0
# %WER 27.83 [ 60025 / 215718, 6628 ins, 12107 del, 41290 sub ] exp/tri5a/decode_test.si/cer_12_0.0
# %WER 17.07 [ 36829 / 215718, 4734 ins, 9938 del, 22157 sub ] exp/nnet3/tdnn_sp/decode_test/cer_12_0.0
#
#

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
# local/nnet3/run_tdnn.sh --stage 8 --train_stage 0	--> bypass feature extreaction 
# local/nnet3/run_tdnn.sh --stage 9			--> decoding only
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
# local/chain/run_tdnn.sh --stage 8			--> bypass feature extraction, if you have already run "local/nnet3/run_tdnn.sh"
# local/chain/run_tdnn.sh --stage 11 --train_stage 120	--> continue the training from iteration 120
local/chain/run_tdnn.sh -stage 8

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
