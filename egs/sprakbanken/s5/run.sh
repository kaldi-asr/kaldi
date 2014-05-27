#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.


# Download the corpus and prepare parallel lists of sound files and text files
# Divide the corpus into train, dev and test sets
local/sprak_data_prep.sh  || exit 1;


# Perform text normalisation of the training set, prepare transcriptions
# Put everything in data/local/dict
#local/sprak_prepare_dict.sh || exit 1;
#local/dict_prep.sh || exit 1;
local/copy_dict.sh || exit 1;

# Repeat text preparation on test set, but do not add to dictionary
test=data/test
python3 local/normalize_transcript_prefixed.py local/norm_dk/numbersUp.tbl $test/text1 $test/onlyids $test/transcripts.am 
local/norm_dk/format_text.sh am $test/transcripts.am > $test/onlytext
paste $test/onlyids $test/onlytext > $test/text


utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang_tmp data/lang || exit 1;

# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc

# Create spk2utt file
utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt

# Extract mfccs 
# p was added to the rspecifier (scp,p:$logdir/wav.JOB.scp) in make_mfcc.sh because some 
# wave files are corrupt 
# Will return a warning message because of the corrupt wave files, but compute them anyway
steps/make_mfcc.sh --nj 30 --cmd $train_cmd data/train exp/make_mfcc/train mfcc 
steps/make_mfcc.sh --nj 30 --cmd $train_cmd data/test exp/make_mfcc/test mfcc 


# Compute cepstral mean and variance normalisation
steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train mfcc && \
steps/compute_cmvn_stats.sh data/test exp/make_mfcc/test mfcc


# Repair data set 
utils/fix_data_dir.sh data/train && utils/fix_data_dir.sh data/test


# Train LM with CMUCLMTK
#local/sprak_train_lm.sh &> data/local/cmuclmtk/lm.log

# Train LM with irstlm
local/train_irstlm.sh data/local/dict/transcripts.txt 3 "b3g" data/lang data/local/trainb3_lm &> data/local/b3g.log &
local/train_irstlm.sh data/local/dict/transcripts.uniq 3 "3g" data/lang data/local/train3_lm &> data/local/3g.log &
#local/train_irstlm.sh data/local/dict/transcripts.txt b4 "b4g" data/lang data/local/trainb4_lm &> data/local/b4g.log &
#local/train_irstlm.sh data/local/dict/transcripts.uniq 4 "4g" data/lang data/local/train4_lm &> data/local/4g.log &

# Make subset with 1k utterances for rapid testing
# Randomly selects 980 utterances from 7 speakers
utils/subset_data_dir.sh --per-spk data/test 140 data/test1k &

# Now make subset with the shortest 120k utterances. 
utils/subset_data_dir.sh --shortest data/train 120000 data/train_120kshort || exit 1;

# Train monophone model on short utterances
steps/train_mono.sh --nj 30 --cmd "$train_cmd" \
  data/train_120kshort data/lang exp/mono0a || exit 1;

# Ensure that LMs are created
wait

utils/mkgraph.sh --mono data/lang_test_3g exp/mono0a exp/mono0a/graph_3g &
utils/mkgraph.sh --mono data/lang_test_b3g exp/mono0a exp/mono0a/graph_b3g &
#utils/mkgraph.sh --mono data/lang_test_4g exp/mono0a exp/mono0a/graph_4g &
#utils/mkgraph.sh --mono data/lang_test_b4g exp/mono0a exp/mono0a/graph_b4g 

# Ensure that all graphs are constructed
wait 



(
steps/decode.sh --nj 7 --cmd "$decode_cmd" \
      exp/mono0a/graph_b3g data/test1k exp/mono0a/decode_b3g_test1k
) &
steps/decode.sh --nj 7 --cmd "$decode_cmd" \
      exp/mono0a/graph_3g data/test1k exp/mono0a/decode_3g_test1k



# steps/align_si.sh --boost-silence 1.25 --nj 42 --cmd "$train_cmd" \
steps/align_si.sh --nj 30 --cmd "$train_cmd" \
   data/train data/lang exp/mono0a exp/mono0a_ali || exit 1;

# steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
steps/train_deltas.sh --cmd "$train_cmd" \
    2000 10000 data/train data/lang exp/mono0a_ali exp/tri1 || exit 1;

wait


utils/mkgraph.sh data/lang_test_3g exp/tri1 exp/tri1/graph_3g &
utils/mkgraph.sh data/lang_test_b3g exp/tri1 exp/tri1/graph_b3g || exit 1;#
 
#(
#steps/decode.sh --nj 7 --cmd "$decode_cmd" \
#  exp/tri1/graph_4g data/test1k exp/tri1/decode_4g_test1k || exit 1;
#) &

(
steps/decode.sh --nj 7 --cmd "$decode_cmd" \
  exp/tri1/graph_3g data/test1k exp/tri1/decode_3g_test1k || exit 1;
) &
steps/decode.sh --nj 7 --cmd "$decode_cmd" \
  exp/tri1/graph_b3g data/test1k exp/tri1/decode_b3g_test1k || exit 1;

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train data/lang exp/tri1 exp/tri1_ali || exit 1;


# Train tri2a, which is deltas + delta-deltas.
steps/train_deltas.sh --cmd "$train_cmd" \
  2500 15000 data/train data/lang exp/tri1_ali exp/tri2a || exit 1;

utils/mkgraph.sh data/lang_test_3g exp/tri2a exp/tri2a/graph_3g || exit 1;

#steps/decode.sh --nj 7 --cmd "$decode_cmd" \
#  exp/tri2a/graph_b3g data/test1k exp/tri2a/decode_b3g_test1k || exit 1;
steps/decode.sh --nj 7 --cmd "$decode_cmd" \
  exp/tri2a/graph_3g data/test1k exp/tri2a/decode_3g_test1k || exit 1;


steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
   2500 15000 data/train data/lang exp/tri1_ali exp/tri2b || exit 1;

utils/mkgraph.sh data/lang_test_3g exp/tri2b exp/tri2b/graph_3g || exit 1;
steps/decode.sh --nj 7 --cmd "$decode_cmd" \
  exp/tri2b/graph_3g data/test1k exp/tri2b/decode_3g_test1k || exit 1;


# Trying Minimum Bayes Risk decoding (like Confusion Network decoding):
mkdir exp/tri2b/decode_3g_test1k_mbr 
cp exp/tri2b/decode_3g_test1k/lat.*.gz exp/tri2b/decode_3g_test1k_mbr 
local/score_mbr.sh --cmd "$decode_cmd" \
 data/test1k/ data/lang_test_3g/ exp/tri2b/decode_3g_test1k_mbr

steps/decode_fromlats.sh --cmd "$decode_cmd" \
  data/test1k data/lang_test_3g exp/tri2b/decode_3g_test1k \
  exp/tri2a/decode_3g_test1k_fromlats || exit 1


# Align tri2b system with si84 data.
steps/align_si.sh  --nj 30 --cmd "$train_cmd" \
  --use-graphs true data/train data/lang exp/tri2b exp/tri2b_ali  || exit 1;

wait


# From 2b system, train 3b which is LDA + MLLT + SAT.
steps/train_sat.sh --cmd "$train_cmd" \
  2500 15000 data/train data/lang exp/tri2b_ali exp/tri3b || exit 1;
utils/mkgraph.sh data/lang_test_3g exp/tri3b exp/tri3b/graph_3g || exit 1;
steps/decode_fmllr.sh --nj 7 --cmd "$decode_cmd" \
  exp/tri3b/graph_3g data/test1k exp/tri3b/decode_3g_test1k || exit 1;


# Trying 4-gram language model
local/train_irstlm.sh data/local/dict/transcripts.uniq 4 "4g" data/lang data/local/train4_lm &> data/local/4g.log
utils/mkgraph.sh data/lang_test_4g exp/tri3b exp/tri3b/graph_4g || exit 1;

steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 7 \
  exp/tri3b/graph_4g data/test1k exp/tri3b/decode_4g_test1k || exit 1;
exit 

# To build RNNLMs uncomment this code

# Repeat text preparation on dev set, but do not add to dictionary
dev=data/dev
python3 local/normalize_transcript_prefixed.py local/norm_dk/numbersUp.tbl $dev/text1 $dev/onlyids $dev/onlytext
local/norm_dk/format_text.sh lm $dev/onlytext > $dev/transcripts.txt
sort -u $dev/transcripts.txt > $dev/transcripts.uniq
local/sprak_train_rnnlms.sh data/local/dict $dev/transcripts.uniq data/local/rnnlms/g_c380_d1k_h100_v130k

# Consumes a lot of memory! Do not run in parallel
local/sprak_run_rnnlms_tri3b.sh data/lang_test_3g data/local/rnnlms/g_c380_d1k_h100_v130k data/test1k exp/tri3b/decode_3g_test1k

# The following two steps, which are a kind of side-branch, try mixing up
# from the 3b system.  This is to demonstrate that script.
 steps/mixup.sh --cmd "$train_cmd" \
   20000 data/train data/lang exp/tri3b exp/tri3b_20k || exit 1;
 steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 7 \
   exp/tri3b/graph_3g data/test1k exp/tri3b_20k/decode_3g_test1k || exit 1;


# From 3b system
steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train data/lang exp/tri3b exp/tri3b_ali || exit 1;

# From 3b system, train another SAT system (tri4a) with all the si284 data.

steps/train_sat.sh  --cmd "$train_cmd" \
  4200 40000 data/train data/lang exp/tri3b_ali exp/tri4a || exit 1;

 utils/mkgraph.sh data/lang_test_3g exp/tri4a exp/tri4a/graph_3g || exit 1;
 steps/decode_fmllr.sh --nj 7 --cmd "$decode_cmd" \
   exp/tri4a/graph_3g data/test1k exp/tri4a/decode_3g_test1k || exit 1;
# steps/decode_fmllr.sh --nj 7 --cmd "$decode_cmd" \
#   exp/tri4a/graph_tgpr data/test_eval92 exp/tri4a/decode_tgpr_eval92 || exit 1;



steps/train_quick.sh --cmd "$train_cmd" \
   4200 40000 data/train data/lang exp/tri3b_ali exp/tri4b || exit 1;

(
 utils/mkgraph.sh data/lang_test_3g exp/tri4b exp/tri4b/graph_3g || exit 1;
 steps/decode_fmllr.sh --nj 7 --cmd "$decode_cmd" \
   exp/tri4b/graph_3g data/test1k exp/tri4b/decode_3g_test1k || exit 1;
) &

 utils/mkgraph.sh data/lang_test_4g exp/tri4b exp/tri4b/graph_4g || exit 1;
 steps/decode_fmllr.sh --nj 7 --cmd "$decode_cmd" \
   exp/tri4b/graph_4g data/test1k exp/tri4b/decode_4g_test1k || exit 1;

wait


# Train and test MMI, and boosted MMI, on tri4b (LDA+MLLT+SAT on
# all the data).  Use 30 jobs.
steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train data/lang exp/tri4b exp/tri4b_ali || exit 1;

## TODO: make sure it runs properly
local/sprak_run_mmi_tri4b.sh 3g test1k

## Works
local/sprak_run_nnet_cpu.sh 3g test1k 

## Works
local/sprak_run_sgmm2.sh test1k

# You probably want to run the hybrid recipe as it is complementary:
#local/run_hybrid.sh


# Getting results [see RESULTS file]
for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done


# KWS setup. We leave it commented out by default
# $duration is the length of the search collection, in seconds
#duration=`feat-to-len scp:data/test_eval92/feats.scp  ark,t:- | awk '{x+=$2} END{print x/100;}'`
#local/generate_example_kws.sh data/test_eval92/ data/kws/
#local/kws_data_prep.sh data/lang_test_bd_tgpr/ data/test_eval92/ data/kws/
#
#steps/make_index.sh --cmd "$decode_cmd" --acwt 0.1 \
#  data/kws/ data/lang_test_bd_tgpr/ \
#  exp/tri4b/decode_bd_tgpr_eval92/ \
#  exp/tri4b/decode_bd_tgpr_eval92/kws
#
#steps/search_index.sh --cmd "$decode_cmd" \
#  data/kws \
#  exp/tri4b/decode_bd_tgpr_eval92/kws
#
# If you want to provide the start time for each utterance, you can use the --segments
# option. In WSJ each file is an utterance, so we don't have to set the start time.
#cat exp/tri4b/decode_bd_tgpr_eval92/kws/result.* | \
#  utils/write_kwslist.pl --flen=0.01 --duration=$duration \
#  --normalize=true --map-utter=data/kws/utter_map \
#  - exp/tri4b/decode_bd_tgpr_eval92/kws/kwslist.xml

# # forward-backward decoding example [way to speed up decoding by decoding forward
# # and backward in time] 
# local/run_fwdbwd.sh

