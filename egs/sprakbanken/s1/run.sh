#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

sprak1=/home/ask/0565-1
sprak2=/home/ask/0565-2
sprak3=/home/ask/0611

parallel=data/local/data/parallel
mkdir -p $parallel/training

local/sprak_data_prep.sh $sprak1 $sprak2 $sprak3 $parallel || exit 1;

local/sprak_prepare_dict.sh || exit 1;

# Repeat text preparation on test set, but do not add to dictionary
python3 local/normalize_transcript.py data/test/text1 data/test/text2 
local/norm_dk/format_text.sh am data/test/text2 > data/test/text

utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang_tmp data/lang || exit 1;


 # We suggest to run the next three commands in the background,
 # as they are not a precondition for the system building and
 # most of the tests: these commands build a dictionary
 # containing many of the OOVs in the WSJ LM training data,
 # and an LM trained directly on that data (i.e. not just
 # copying the arpa files from the disks from LDC).
 # Caution: the commands below will only work if $decode_cmd 
 # is setup to use qsub.  Else, just remove the --cmd option.
 # NOTE: If you have a setup corresponding to the cstr_wsj_data_prep.sh style,
 # use local/cstr_wsj_extend_dict.sh $corpus/wsj1/doc/ instead.

 # Note: I am commenting out the RNNLM-building commands below.  They take up a lot
 # of CPU time and are not really part of the "main recipe."
 # Be careful: appending things like "-l mem_free=10G" to $decode_cmd
 # won't always work, it depends what $decode_cmd is.
 # (
 #  local/wsj_extend_dict.sh $wsj1/13-32.1  && \
 #  utils/prepare_lang.sh data/local/dict_larger "<SPOKEN_NOISE>" data/local/lang_larger data/lang_bd && \
 #  local/wsj_train_lms.sh &&
 #  local/wsj_format_local_lms.sh # &&
 #
 #   (  local/wsj_train_rnnlms.sh --cmd "$decode_cmd -l mem_free=10G" data/local/rnnlm.h30.voc10k &
 #       sleep 20; # wait till tools compiled.
 #     local/wsj_train_rnnlms.sh --cmd "$decode_cmd -l mem_free=12G" \
 #      --hidden 100 --nwords 20000 --class 350 --direct 1500 data/local/rnnlm.h100.voc20k &
 #     local/wsj_train_rnnlms.sh --cmd "$decode_cmd -l mem_free=14G" \
 #      --hidden 200 --nwords 30000 --class 350 --direct 1500 data/local/rnnlm.h200.voc30k &
 #     local/wsj_train_rnnlms.sh --cmd "$decode_cmd -l mem_free=16G" \
 #      --hidden 300 --nwords 40000 --class 400 --direct 2000 data/local/rnnlm.h300.voc40k &
 #   )
 # ) &


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
steps/make_mfcc.sh --nj 50 --cmd $train_cmd data/train exp/make_mfcc/train mfcc 
steps/make_mfcc.sh --nj 50 --cmd $train_cmd data/test exp/make_mfcc/test mfcc 


# Compute cepstral mean and variance normalisation
steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train mfcc && \
steps/compute_cmvn_stats.sh data/test exp/make_mfcc/test mfcc


# Repair data set 
utils/fix_data_dir.sh data/train && utils/fix_data_dir.sh data/test


# Train LM with CMUCLMTK
#local/sprak_train_lm.sh &> data/local/cmuclmtk/lm.log

# Train LM with irstlm
local/train_irstlm.sh data/local/dict/transcripts2 3 "3g" data/lang data/local/train3_lm &> data/local/lm.log &

local/train_irstlm.sh data/local/dict/transcripts2 4 "4g" data/lang data/local/train4_lm &> data/local/lm.log &


# Make subset with 1k utterances for quick testing
# Randomly selects 980 utterances from 7 speakers
utils/subset_data_dir.sh --per-spk data/test 140 data/test1k &

# Now make subset with the shortest 120k utterances. 
utils/subset_data_dir.sh --shortest data/train 120000 data/train_120kshort || exit 1;



# Note: the --boost-silence option should probably be omitted by default
# for normal setups.  It doesn't always help. [it's to discourage non-silence
# models from modeling silence.]
#steps/train_mono.sh --boost-silence 1.25 --nj 42 --cmd "$train_cmd" \
steps/train_mono.sh --nj 50 --cmd "$train_cmd" \
  data/train_120kshort data/lang exp/mono0a || exit 1;

# Ensure that LMs are created
wait

utils/mkgraph.sh --mono data/lang_test_3g exp/mono0a exp/mono0a/graph_3g &
utils/mkgraph.sh --mono data/lang_test_4g exp/mono0a exp/mono0a/graph_4g 

# Ensure that both graphs are constructed
wait 

(
steps/decode.sh --nj 7 --cmd "$decode_cmd" \
      exp/mono0a/graph_4g data/test1k exp/mono0a/decode_4g_test1k
) &
steps/decode.sh --nj 7 --cmd "$decode_cmd" \
      exp/mono0a/graph_3g data/test1k exp/mono0a/decode_3g_test1k



# steps/align_si.sh --boost-silence 1.25 --nj 42 --cmd "$train_cmd" \
steps/align_si.sh --nj 50 --cmd "$train_cmd" \
   data/train data/lang exp/mono0a exp/mono0a_ali || exit 1;

# steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
steps/train_deltas.sh --cmd "$train_cmd" \
    2000 10000 data/train data/lang exp/mono0a_ali exp/tri1 || exit 1;

wait

utils/mkgraph.sh data/lang_test_3g exp/tri1 exp/tri1/graph_3g &
utils/mkgraph.sh data/lang_test_4g exp/tri1 exp/tri1/graph_4g || exit 1;#
 
#(
#steps/decode.sh --nj 7 --cmd "$decode_cmd" \
#  exp/tri1/graph_4g data/test exp/tri1/decode_4g_test || exit 1;
#) &

steps/decode.sh --nj 7 --cmd "$decode_cmd" \
  exp/tri1/graph_3g data/test1k exp/tri1/decode_3g_test1k || exit 1;

steps/align_si.sh --nj 50 --cmd "$train_cmd" \
  data/train data/lang exp/tri1 exp/tri1_ali || exit 1;


# Train tri2a, which is deltas + delta-deltas, on si84 data.
steps/train_deltas.sh --cmd "$train_cmd" \
  2500 15000 data/train data/lang exp/tri1_ali exp/tri2a || exit 1;

utils/mkgraph.sh data/lang_test_3g exp/tri2a exp/tri2a/graph_3g || exit 1;

#steps/decode.sh --nj 7 --cmd "$decode_cmd" \
#  exp/tri2a/graph_3g data/test exp/tri2a/decode_3g_test || exit 1;
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
steps/align_si.sh  --nj 50 --cmd "$train_cmd" \
  --use-graphs true data/train data/lang exp/tri2b exp/tri2b_ali  || exit 1;

wait


# From 2b system, train 3b which is LDA + MLLT + SAT.
steps/train_sat.sh --cmd "$train_cmd" \
  2500 15000 data/train data/lang exp/tri2b_ali exp/tri3b || exit 1;
utils/mkgraph.sh data/lang_test_3g exp/tri3b exp/tri3b/graph_3g || exit 1;
steps/decode_fmllr.sh --nj 7 --cmd "$decode_cmd" \
  exp/tri3b/graph_3g data/test1k exp/tri3b/decode_3g_test1k || exit 1;


# Trying 4-gram language model
utils/mkgraph.sh data/lang_test_4g exp/tri3b exp/tri3b/graph_4g || exit 1;

steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 7 \
  exp/tri3b/graph_4g data/test1k exp/tri3b/decode_4g_test1k || exit 1;


# The command below is commented out as we commented out the steps above
# that build the RNNLMs, so it would fail.
# local/run_rnnlms_tri3b.sh

# The following two steps, which are a kind of side-branch, try mixing up
# from the 3b system.  This is to demonstrate that script.
 steps/mixup.sh --cmd "$train_cmd" \
   20000 data/train data/lang exp/tri3b exp/tri3b_20k || exit 1;
 steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 7 \
   exp/tri3b/graph_3g data/test1k exp/tri3b_20k/decode_3g_test1k || exit 1;



# From 3b system, align all si284 data.
steps/align_fmllr.sh --nj 50 --cmd "$train_cmd" \
  data/train data/lang exp/tri3b exp/tri3b_ali || exit 1;

# From 3b system, train another SAT system (tri4a) with all the si284 data.

steps/train_sat.sh  --cmd "$train_cmd" \
  4200 40000 data/train data/lang exp/tri3b_ali exp/tri4a || exit 1;
(
 utils/mkgraph.sh data/lang_test_3g exp/tri4a exp/tri4a/graph_3g || exit 1;
 steps/decode_fmllr.sh --nj 7 --cmd "$decode_cmd" \
   exp/tri4a/graph_3g data/test1k exp/tri4a/decode_3g_test1k || exit 1;
# steps/decode_fmllr.sh --nj 7 --cmd "$decode_cmd" \
#   exp/tri4a/graph_tgpr data/test_eval92 exp/tri4a/decode_tgpr_eval92 || exit 1;
) & 


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
steps/align_fmllr.sh --nj 50 --cmd "$train_cmd" \
  data/train data/lang exp/tri4b exp/tri4b_ali || exit 1;

local/extra_run_mmi_tri4b.sh 3g test1k

local/run_nnet_cpu.sh

## Segregated some SGMM builds into a separate file.
#local/run_sgmm.sh

# You probably want to run the sgmm2 recipe as it's generally a bit better:
local/extra_run_sgmm2.sh 3g test1k

# You probably wany to run the hybrid recipe as it is complementary:
#local/run_hybrid.sh


# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done


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

