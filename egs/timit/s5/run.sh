#!/bin/bash

# Copyright 2013  Bagher BabaAli

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
[ -f path.sh ] && . ./path.sh

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

timit=/export/corpora5/LDC/LDC93S1/timit/TIMIT

local/timit_data_prep.sh $timit  || exit 1;

local/timit_prepare_dict.sh || exit 1;

utils/prepare_lang.sh --position-dependent-phones false --num-sil-states 3 \
 data/local/dict "sil" data/local/lang_tmp data/lang || exit 1;

local/timit_format_data.sh || exit 1;

# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc
for x in test train; do 
 steps/make_mfcc.sh --cmd "$train_cmd" --nj 30 \
   data/$x exp/make_mfcc/$x $mfccdir || exit 1;
 steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
done

# Note: the --boost-silence option should probably be omitted by default
# for normal setups.  It doesn't always help. [it's to discourage non-silence
# models from modeling silence.]
steps/train_mono.sh --boost-silence 1.25 --nj 30 --cmd "$train_cmd" \
  data/train data/lang exp/mono0a || exit 1;

 utils/mkgraph.sh --mono data/lang_test_bg exp/mono0a exp/mono0a/graph_bg && \
 steps/decode.sh --nj 30  --cmd "$decode_cmd" \
      exp/mono0a/graph_bg data/test exp/mono0a/decode_bg_test 

steps/align_si.sh --boost-silence 1.25 --nj 30 --cmd "$train_cmd" \
   data/train data/lang exp/mono0a exp/mono0a_ali || exit 1;

steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/train data/lang exp/mono0a_ali exp/tri1 || exit 1;

utils/mkgraph.sh data/lang_test_bg exp/tri1 exp/tri1/graph_bg || exit 1;

steps/decode.sh --nj 30 --cmd "$decode_cmd" \
  exp/tri1/graph_bg data/test exp/tri1/decode_bg_test || exit 1;

# demonstrate how to get lattices that are "word-aligned" (arcs coincide with
# words, with boundaries in the right place).
#sil_label=`grep '!SIL' data/lang_test_bg/words.txt | awk '{print $2}'`
#steps/word_align_lattices.sh --cmd "$train_cmd" --silence-label $sil_label \
#  data/lang_test_bg exp/tri1/decode_bg_test exp/tri1/decode_bg_test_aligned || exit 1;

# Align tri1 system with train data.
steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train data/lang exp/tri1 exp/tri1_ali_train || exit 1;

# Train tri2a, which is deltas + delta-deltas, on train data.
steps/train_deltas.sh --cmd "$train_cmd" \
  2500 15000 data/train data/lang exp/tri1_ali_train exp/tri2a || exit 1;

utils/mkgraph.sh data/lang_test_bg exp/tri2a exp/tri2a/graph_bg || exit 1;

steps/decode.sh --nj 30 --cmd "$decode_cmd" \
  exp/tri2a/graph_bg data/test exp/tri2a/decode_bg_test || exit 1;

steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
   2500 15000 data/train data/lang exp/tri1_ali_train exp/tri2b || exit 1;

utils/mkgraph.sh data/lang_test_bg exp/tri2b exp/tri2b/graph_bg || exit 1;
steps/decode.sh --nj 30 --cmd "$decode_cmd" \
  exp/tri2b/graph_bg data/test exp/tri2b/decode_bg_test || exit 1;

# Trying Minimum Bayes Risk decoding (like Confusion Network decoding):
mkdir exp/tri2b/decode_bg_test_mbr 
cp exp/tri2b/decode_bg_test/lat.*.gz exp/tri2b/decode_bg_test_mbr 
local/score_mbr.sh --cmd "$decode_cmd" \
 data/test/ data/lang_test_bg/ exp/tri2b/decode_bg_test_mbr

steps/decode_fromlats.sh --cmd "$decode_cmd" \
  data/test data/lang_test_bg exp/tri2b/decode_bg_test \
  exp/tri2a/decode_bg_test_fromlats || exit 1;

# Align tri2b system with train data.
steps/align_si.sh  --nj 30 --cmd "$train_cmd" \
  --use-graphs true data/train data/lang exp/tri2b exp/tri2b_ali_train  || exit 1;

local/run_mmi_tri2b.sh

# From 2b system, train 3b which is LDA + MLLT + SAT.
steps/train_sat.sh --cmd "$train_cmd" \
  2500 15000 data/train data/lang exp/tri2b_ali_train exp/tri3b || exit 1;
utils/mkgraph.sh data/lang_test_bg exp/tri3b exp/tri3b/graph_bg || exit 1;
steps/decode_fmllr.sh --nj 30 --cmd "$decode_cmd" \
  exp/tri3b/graph_bg data/test exp/tri3b/decode_bg_test || exit 1;

# At this point you could run the command below; this gets
# results that demonstrate the basis-fMLLR adaptation (adaptation
# on small amounts of adaptation data).
local/run_basis_fmllr.sh


# Train and test MMI, and boosted MMI, on tri3b (LDA+MLLT+SAT on
# all the data).  Use 30 jobs.
# From 3b system, align all train data.
steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train data/lang exp/tri3b exp/tri3b_ali_train || exit 1;

local/run_mmi_tri3b.sh

# You probably want to run the sgmm2 recipe as it's generally a bit better:
local/run_sgmm2.sh

# You probably wany to run the hybrid recipe as it is complementary:
#local/run_hybrid.sh


# Getting results [see RESULTS file]
 for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done

 exit 1;

# end



