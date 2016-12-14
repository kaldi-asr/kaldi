#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh # so python3 is on the path if not on the system (we made a link to utils/).a

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.


# Download the corpus and prepare parallel lists of sound files and text files
# Divide the corpus into train, dev and test sets
local/sprak_data_prep.sh  || exit 1;

# Perform text normalisation, prepare dict folder and LM data transcriptions
# This setup uses previsously prepared data. eSpeak must be installed and in PATH to use dict_prep.sh
#local/dict_prep.sh || exit 1;
local/copy_dict.sh || exit 1;


utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang_tmp data/lang || exit 1;

# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc


# Extract mfccs 
# p was added to the rspecifier (scp,p:$logdir/wav.JOB.scp) in make_mfcc.sh because some 
# wave files are corrupt 
# Will return a warning message because of the corrupt audio files, but compute them anyway
# If this step fails and prints a partial diff, rerun from sprak_data_prep.sh

steps/make_mfcc.sh --nj 10 --cmd $train_cmd data/test exp/make_mfcc/test mfcc &
steps/make_mfcc.sh --nj 10 --cmd $train_cmd data/dev exp/make_mfcc/dev mfcc &
steps/make_mfcc.sh --nj 10 --cmd $train_cmd data/train exp/make_mfcc/train mfcc || exit 1;
wait

# Compute cepstral mean and variance normalisation
steps/compute_cmvn_stats.sh data/test exp/make_mfcc/test mfcc &
steps/compute_cmvn_stats.sh data/dev exp/make_mfcc/dev mfcc &
steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train mfcc 

wait

# Repair data set (remove corrupt data points with corrupt audio)

utils/fix_data_dir.sh data/test &
utils/fix_data_dir.sh data/dev &
utils/fix_data_dir.sh data/train 
wait

# Train LM with CMUCLMTK
# This setup uses IRSTLM
#local/sprak_train_lm.sh &> data/local/cmuclmtk/lm.log

# Train LM with irstlm
local/train_irstlm.sh data/local/transcript_lm/transcripts.uniq 3 "3g" data/lang data/local/train3_lm &> data/local/3g.log &
local/train_irstlm.sh data/local/transcript_lm/transcripts.uniq 4 "4g" data/lang data/local/train4_lm &> data/local/4g.log 

# Make subset with 1k utterances for rapid testing
# Randomly selects 980 utterances from 7 speakers
utils/subset_data_dir.sh --per-spk data/test 140 data/test1k &

# Now make subset of the training data with the shortest 120k utterances. 
utils/subset_data_dir.sh --shortest data/train 120000 data/train_120kshort || exit 1;

# Train monophone model on short utterances
steps/train_mono.sh --nj 30 --cmd "$train_cmd" \
  data/train_120kshort data/lang exp/mono0a || exit 1;

# Ensure that LMs are created
wait

utils/mkgraph.sh data/lang_test_3g exp/mono0a exp/mono0a/graph_3g &
utils/mkgraph.sh data/lang_test_4g exp/mono0a exp/mono0a/graph_4g &

# Ensure that all graphs are constructed
wait 

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
utils/mkgraph.sh data/lang_test_4g exp/tri1 exp/tri1/graph_4g || exit 1;
 
(
steps/decode.sh --nj 7 --cmd "$decode_cmd" \
  exp/tri1/graph_4g data/test1k exp/tri1/decode_4g_test1k || exit 1;
) &

(
steps/decode.sh --nj 7 --cmd "$decode_cmd" \
  exp/tri1/graph_3g data/test1k exp/tri1/decode_3g_test1k || exit 1;
) &

wait

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train data/lang exp/tri1 exp/tri1_ali || exit 1;


# Train tri2a, which is deltas + delta-deltas.
steps/train_deltas.sh --cmd "$train_cmd" \
  2500 15000 data/train data/lang exp/tri1_ali exp/tri2a || exit 1;

utils/mkgraph.sh data/lang_test_3g exp/tri2a exp/tri2a/graph_3g || exit 1;

steps/decode.sh --nj 7 --cmd "$decode_cmd" \
  exp/tri2a/graph_3g data/test1k exp/tri2a/decode_3g_test1k || exit 1;


steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=5 --right-context=5" \
   2500 15000 data/train data/lang exp/tri1_ali exp/tri2b || exit 1;

utils/mkgraph.sh data/lang_test_3g exp/tri2b exp/tri2b/graph_3g || exit 1;
steps/decode.sh --nj 7 --cmd "$decode_cmd" \
  exp/tri2b/graph_3g data/test1k exp/tri2b/decode_3g_test1k || exit 1;


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
utils/mkgraph.sh data/lang_test_4g exp/tri3b exp/tri3b/graph_4g || exit 1;

steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 7 \
  exp/tri3b/graph_4g data/test1k exp/tri3b/decode_4g_test1k || exit 1;

# This is commented out for now as it's not important for the main recipe.
## Train RNN for reranking
#local/sprak_train_rnnlms.sh data/local/dict data/dev/transcripts.uniq data/local/rnnlms/g_c380_d1k_h100_v130k
## Consumes a lot of memory! Do not run in parallel
#local/sprak_run_rnnlms_tri3b.sh data/lang_test_3g data/local/rnnlms/g_c380_d1k_h100_v130k data/test1k exp/tri3b/decode_3g_test1k


# From 3b system
steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train data/lang exp/tri3b exp/tri3b_ali || exit 1;

# From 3b system, train another SAT system (tri4a) with all the si284 data.

steps/train_sat.sh  --cmd "$train_cmd" \
  4200 40000 data/train data/lang exp/tri3b_ali exp/tri4a || exit 1;

utils/mkgraph.sh data/lang_test_3g exp/tri4a exp/tri4a/graph_3g || exit 1;
steps/decode_fmllr.sh --nj 7 --cmd "$decode_cmd" \
   exp/tri4a/graph_3g data/test1k exp/tri4a/decode_3g_test1k || exit 1;


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

# alignment used to train nnets and sgmms
steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train data/lang exp/tri4b exp/tri4b_ali || exit 1;

## Works
local/sprak_run_nnet_cpu.sh 3g test1k 

## Works
local/sprak_run_sgmm2.sh test1k


# Getting results [see RESULTS file]
for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
