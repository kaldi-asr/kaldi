#!/bin/bash

# Copyright 2010-2011 Microsoft Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

exit 1;
# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.
# Caution: some of the graph creation steps use quite a bit of memory, so you



# Data prep
local/swbd_p1_data_prep.sh /mnt/matylda2/data/SWITCHBOARD_1R2

local/swbd_p1_format_data.sh

# Data preparation and formatting for eval2000 (note: the "text" file
# is not very much preprocessed; for actual WER reporting we'll use
# sclite.
local/eval2000_data_prep.sh /mnt/matylda2/data/HUB5_2000/ /mnt/matylda2/data/HUB5_2000/

# mfccdir should be some place with a largish disk where you
# want to store MFCC features. 
#mfccdir=/mnt/matylda6/ijanda/kaldi_swbd_mfcc
mfccdir=/mnt/matylda6/jhu09/qpovey/kaldi_swbd_mfcc
cmd="queue.pl -q all.q@@blade" # remove the option if no queue.
local/make_mfcc_segs.sh --num-jobs 10 --cmd "$cmd" data/train exp/make_mfcc/train $mfccdir
# after this, the next command will remove the small number of utterances
# that couldn't be extracted for some reason (e.g. too short; no such file).
scripts/fix_data_dir.sh data/train

local/make_mfcc_segs.sh --num-jobs 4 data/eval2000 exp/make_mfcc/eval2000 $mfccdir
scripts/fix_data_dir.sh data/eval2000

# Now-- there are 264k utterances, and we want to start the monophone training
# on relatively short utterances (easier to align), but not only the very shortest
# ones (mostly uh-huh).  So take the 100k shortest ones, and then take 10k random
# utterances from those.
scripts/subset_data_dir.sh --shortest data/train 100000 data/train_100kshort
scripts/subset_data_dir.sh  data/train_100kshort 10000 data/train_10k
local/remove_dup_utts.sh 100 data/train_10k data/train_10k_nodup

local/remove_dup_utts.sh 200 data/train_100kshort data/train_100k_nodup

local/remove_dup_utts.sh 300 data/train data/train_nodup

( . path.sh; 
  # make sure mfccdir is defined as above..
  cp data/train_10k_nodup/feats.scp{,.bak} 
  copy-feats scp:data/train_10k_nodup/feats.scp  ark,scp:$mfccdir/kaldi_swbd_10k_nodup.ark,$mfccdir/kaldi_swbd_10k_nodup.scp \
  && cp $mfccdir/kaldi_swbd_10k_nodup.scp data/train_10k_nodup/feats.scp
)

( . path.sh; 
  # make sure mfccdir is defined as above..
  cp data/train_100k_nodup/feats.scp{,.bak} 
  copy-feats scp:data/train_100k_nodup/feats.scp  ark,scp:$mfccdir/kaldi_swbd_100k_nodup.ark,$mfccdir/kaldi_swbd_100k_nodup.scp \
  && cp $mfccdir/kaldi_swbd_100k_nodup.scp data/train_100k_nodup/feats.scp
)
 


decode_cmd="queue.pl -q all.q@@blade -l ram_free=1200M,mem_free=1200M"
train_cmd="queue.pl -q all.q@@blade -l ram_free=700M,mem_free=700M"

steps/train_mono.sh --num-jobs 10 --cmd "$train_cmd" \
  data/train_10k_nodup data/lang exp/mono0a

steps/align_deltas.sh --num-jobs 30 --cmd "$train_cmd" \
   data/train_100k_nodup data/lang exp/mono0a exp/mono0a_ali

steps/train_deltas.sh --num-jobs 30 --cmd "$train_cmd" \
    4000 20000 data/train_100k_nodup data/lang exp/mono0a_ali exp/tri1

steps/align_deltas.sh --num-jobs 30 --cmd "$train_cmd" \
   data/train_100k_nodup data/lang exp/tri1 exp/tri1_ali

steps/train_deltas.sh --num-jobs 30 --cmd "$train_cmd" \
    4000 20000 data/train_100k_nodup data/lang exp/tri1_ali exp/tri2

steps/align_deltas.sh --num-jobs 30 --cmd "$train_cmd" \
   data/train_100k_nodup data/lang exp/tri2 exp/tri2_ali

# Train tri3a, which is LDA+MLLT, on 100k_nodup data.
steps/train_lda_mllt.sh --num-jobs 30 --cmd "$train_cmd" \
   4000 20000 data/train_100k_nodup data/lang exp/tri2_ali exp/tri3a

steps/align_lda_mllt.sh  --num-jobs 30 --cmd "$train_cmd" \
  --use-graphs data/train_100k_nodup data/lang exp/tri3a exp/tri3a_ali

steps/train_lda_mllt_sat.sh  --num-jobs 30 --cmd "$train_cmd" \
  4000 20000 data/train_100k_nodup data/lang exp/tri3a_ali exp/tri4a

scripts/mkgraph.sh data/lang_test exp/tri4a exp/tri4a/graph
scripts/decode.sh -l data/lang_test --num-jobs 10 --cmd "$decode_cmd" \
 steps/decode_lda_mllt_sat.sh exp/tri4a/graph data/eval2000 exp/tri4a/decode_eval2000

steps/align_lda_mllt_sat.sh  --num-jobs 30 --cmd "$train_cmd" \
  data/train_nodup data/lang exp/tri4a exp/tri4a_ali_all_nodup

# Note: up to this point we probably had too many leaves.
steps/train_lda_mllt_sat.sh  --num-jobs 30 --cmd "$train_cmd" \
  4000 150000 data/train_nodup data/lang exp/tri4a_ali_all_nodup exp/tri5a

scripts/mkgraph.sh data/lang_test exp/tri5a exp/tri5a/graph
scripts/decode.sh -l data/lang_test --num-jobs 10 --cmd "$decode_cmd" \
  steps/decode_lda_mllt_sat.sh exp/tri5a/graph data/eval2000 exp/tri5a/decode_eval2000

# Align the 5a system; we'll train an SGMM system on top of 
# LDA+MLLT+SAT, and use 5a system for 1st pass.
steps/align_lda_mllt_sat.sh  --num-jobs 30 --cmd "$train_cmd" \
  data/train_nodup data/lang exp/tri5a exp/tri5a_ali_all_nodup

steps/train_ubm_lda_etc.sh --num-jobs 30 --cmd "$train_cmd" \
  700 data/train_nodup data/lang exp/tri5a_ali_all_nodup exp/ubm6a
steps/train_sgmm_lda_etc.sh --num-jobs 30 --cmd "$train_cmd" \
   4500 40000 41 40 data/train_nodup data/lang exp/tri5a_ali_all_nodup exp/ubm6a/final.ubm exp/sgmm6a
scripts/mkgraph.sh data/lang_test_tgpr exp/sgmm6a exp/sgmm6a/graph_tgpr
# have to match num-jobs with 5a decode.
scripts/decode.sh --num-jobs 10 --cmd "$decode_cmd" steps/decode_sgmm_lda_etc.sh \
   exp/sgmm6a/graph_tgpr data/eval2000 exp/sgmm6a/decode_eval2000 exp/tri5a/decode_eval2000


for x in exp/*/decode_*; do [ -d $x ] && grep Mean  $x/score_*/*.sys | scripts/best_wer.sh; done