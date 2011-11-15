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

# Note: this is work in progress!  This will be the new, "cleaner" version
# of the WSJ scripts.

exit 1;
# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.
# Caution: some of the graph creation steps use quite a bit of memory, so you
# might want to run this script on a machine that has plenty of memory.


# The next command line is an example; you have to give the script
# command-line arguments corresponding to the WSJ disks from LDC.  
# Another example set of command line arguments is
# /ais/gobi2/speech/WSJ/*/??-{?,??}.?
#  These must be absolute,  not relative, pathnames.
local/wsj_data_prep.sh /mnt/matylda2/data/WSJ?/??-{?,??}.?

local/wsj_prepare_dict.sh

local/wsj_format_data.sh

# We suggest to run the next three commands in the background,
# as they are not a precondition for the system building and
# most of the tests: these commands build a dictionary
# containing many of the OOVs in the WSJ LM training data,
# and an LM trained directly on that data (i.e. not just
# copying the arpa files from the disks from LDC).
(
 local/wsj_extend_dict.sh /mnt/matylda2/data/WSJ1/13-32.1  && \
 local/wsj_prepare_local_dict.sh && \
 local/wsj_train_lms.sh && \
 local/wsj_format_data_local.sh
) &


# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=/mnt/matylda6/jhu09/qpovey/kaldi_wsj_mfcc
for x in test_eval92 test_eval93 test_dev93 train_si284; do 
 steps/make_mfcc.sh data/$x exp/make_mfcc/$x $mfccdir 4
done


mkdir data/train_si84
for x in feats.scp text utt2spk wav.scp; do
  head -7138 data/train_si284/$x > data/train_si84/$x
done
scripts/utt2spk_to_spk2utt.pl data/train_si84/utt2spk > data/train_si84/spk2utt
scripts/filter_scp.pl data/train_si84/spk2utt data/train_si284/spk2gender > data/train_si84/spk2gender

# Now make subset with the shortest 2k utterances from si-84.
scripts/subset_data_dir.sh data/train_si84 2000 data/train_si84_2kshort

# Now make subset with half of the data from si-84.
scripts/subset_data_dir.sh data/train_si84 3500 data/train_si84_half

# you can change these commands to just run.pl to make them run
# locally, but in that case you should change the num-jobs to
# the #cpus on your machine or fewer.
decode_cmd="queue.pl -q all.q@@blade -l ram_free=1200M,mem_free=1200M"
train_cmd="queue.pl -q all.q@@blade -l ram_free=700M,mem_free=700M"

steps/train_mono.sh --num-jobs 10 --cmd "$train_cmd" \
  data/train_si84_2kshort data/lang exp/mono0a

(
scripts/mkgraph.sh --mono data/lang_test_fgpr exp/mono0a exp/mono0a/graph_fgpr
scripts/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh exp/mono0a/graph_tgpr data/test_dev93 exp/mono0a/decode_tgpr_dev93
scripts/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh exp/mono0a/graph_tgpr data/test_eval92 exp/mono0a/decode_tgpr_eval92
)&

# This queue option will be supplied to all alignment
# and training scripts.  Note: you have to supply the same num-jobs
# to the alignment and training scripts, as the archives are split
# up in this way.


steps/align_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
   data/train_si84_half data/lang exp/mono0a exp/mono0a_ali

steps/train_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
    2000 10000 data/train_si84_half data/lang exp/mono0a_ali exp/tri1

scripts/mkgraph.sh data/lang_test_tgpr exp/tri1 exp/tri1/graph_tgpr

scripts/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh exp/tri1/graph_tgpr data/test_dev93 exp/tri1/decode_tgpr_dev93

# test various modes of LM rescoring (4 is the default one).
for mode in 1 2 3 4; do
scripts/lmrescore.sh --mode $mode --cmd "$decode_cmd" data/lang_test_{tgpr,tg} \
  data/test_dev93 exp/tri1/decode_tgpr_dev93 exp/tri1/decode_tgpr_dev93_tg$mode 
done

# Align tri1 system with si84 data.
steps/align_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri1 exp/tri1_ali_si84


# Train tri2a, which is deltas + delta-deltas, on si84 data.
steps/train_deltas.sh  --num-jobs 10 --cmd "$train_cmd" \
  2500 15000 data/train_si84 data/lang exp/tri1_ali_si84 exp/tri2a
scripts/mkgraph.sh data/lang_test_tgpr exp/tri2a exp/tri2a/graph_tgpr
scripts/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh exp/tri2a/graph_tgpr data/test_dev93 exp/tri2a/decode_tgpr_dev93


# Train tri2b, which is LDA+MLLT, on si84 data.
steps/train_lda_mllt.sh --num-jobs 10 --cmd "$train_cmd" \
   2500 15000 data/train_si84 data/lang exp/tri1_ali_si84 exp/tri2b
scripts/mkgraph.sh data/lang_test_tgpr exp/tri2b exp/tri2b/graph_tgpr
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt.sh exp/tri2b/graph_tgpr data/test_eval92 exp/tri2b/decode_tgpr_eval92
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt.sh exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b/decode_tgpr_dev93

# Now, with dev93, compare lattice rescoring with biglm decoding,
# going from tgpr to tg.  Note: results are not the same, even though they should
# be, and I believe this is due to the beams not being wide enough.  The pruning
# seems to be a bit too narrow in the current scripts (got at least 0.7% absolute
# improvement from loosening beams from their current values).

# Note: if you are running this soon after it's created and not from scratch, you
# may have to rerun local/wsj_format_data.sh before the command below (a rmepsilon
# stage in there that's necessary).
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_biglm.sh exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b/decode_tgpr_dev93_tg_biglm data/lang_test_tgpr/G.fst data/lang_test_tg/G.fst
# baseline via LM rescoring of lattices.
scripts/lmrescore.sh --cmd "$decode_cmd" data/lang_test_tgpr/ data/lang_test_tg/ \
  data/test_dev93 exp/tri2b/decode_tgpr_dev93 exp/tri2b/decode_tgpr_dev93_tg



# Demonstrate 'cross-tree' lattice rescoring where we create utterance-specific
# decoding graphs from one system's lattices and rescore with another system.
# Note: we could easily do this with the trigram LM, unpruned, but for comparability
# with the experiments above we do it with the pruned one.
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_fromlats.sh data/lang_test_tgpr data/test_dev93 exp/tri2b/decode_tgpr_dev93_fromlats exp/tri2a/decode_tgpr_dev93

# Align tri2b system with si84 data.
steps/align_lda_mllt.sh  --num-jobs 10 --cmd "$train_cmd" \
  --use-graphs data/train_si84 data/lang exp/tri2b exp/tri2b_ali_si84

# Train and test MMI (and boosted MMI) on tri2b system.
steps/make_denlats_lda_etc.sh --num-jobs 10 --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri2b_denlats_si84
steps/train_lda_etc_mmi.sh --num-jobs 10  --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri2b_denlats_si84 exp/tri2b exp/tri2b_mmi
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt.sh exp/tri2b/graph_tgpr data/test_eval92 exp/tri2b_mmi/decode_tgpr_eval92
steps/train_lda_etc_mmi.sh --num-jobs 10 --boost 0.1 --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri2b_denlats_si84 exp/tri2b exp/tri2b_mmi_b0.1
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt.sh exp/tri2b/graph_tgpr data/test_eval92 exp/tri2b_mmi_b0.1/decode_tgpr_eval92

# Train LDA+ET system.
steps/train_lda_et.sh --num-jobs 10 --cmd "$train_cmd" \
  2500 15000 data/train_si84 data/lang exp/tri1_ali_si84 exp/tri2c
scripts/mkgraph.sh data/lang_test_tgpr exp/tri2c exp/tri2c/graph_tgpr
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_et.sh exp/tri2c/graph_tgpr data/test_dev93 exp/tri2c/decode_tgpr_dev93
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_et_2pass.sh exp/tri2c/graph_tgpr data/test_dev93 exp/tri2c/decode_tgpr_dev93_2pass

# From 2b system, train 3b which is LDA + MLLT + SAT.
steps/train_lda_mllt_sat.sh  --num-jobs 10 --cmd "$train_cmd" \
  2500 15000 data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri3b
scripts/mkgraph.sh data/lang_test_tgpr exp/tri3b exp/tri3b/graph_tgpr
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh exp/tri3b/graph_tgpr \
  data/test_dev93 exp/tri3b/decode_tgpr_dev93
scripts/lmrescore.sh --cmd "$decode_cmd" data/lang_test_tgpr data/lang_test_tg \
  data/test_dev93 exp/tri3b/decode_tgpr_dev93 exp/tri3b/decode_tgpr_dev93_tg
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh exp/tri3b/graph_tgpr \
  data/test_eval92 exp/tri3b/decode_tgpr_eval92
scripts/lmrescore.sh --cmd "$decode_cmd" data/lang_test_tgpr data/lang_test_tg \
  data/test_eval92 exp/tri3b/decode_tgpr_eval92 exp/tri3b/decode_tgpr_eval92_tg

# Trying the larger dictionary ("big-dict"/bd) + locally produced LM.
scripts/mkgraph.sh data/lang_test_bd_tgpr exp/tri3b exp/tri3b/graph_bd_tgpr
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh exp/tri3b/graph_bd_tgpr \
  data/test_eval92 exp/tri3b/decode_bd_tgpr_eval92

# From 3b system, align all si284 data.
steps/align_lda_mllt_sat.sh --num-jobs 10 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri3b exp/tri3b_ali_si284

steps/train_lda_etc_quick.sh --num-jobs 10 --cmd "$train_cmd" \
   4200 40000 data/train_si284 data/lang exp/tri3b_ali_si284 exp/tri4b
scripts/mkgraph.sh data/lang_test_tgpr exp/tri4b exp/tri4b/graph_tgpr
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh exp/tri4b/graph_tgpr data/test_dev93 exp/tri4b/decode_tgpr_dev93
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh exp/tri4b/graph_tgpr data/test_eval92 exp/tri4b/decode_tgpr_eval92

# Train and test MMI, and boosted MMI, on tri4b.
steps/align_lda_mllt_sat.sh --num-jobs 30 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri4b exp/tri4b_ali_si284
steps/make_denlats_lda_etc.sh --num-jobs 30 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri4b_ali_si284 exp/tri4b_denlats_si284
steps/train_lda_etc_mmi.sh --num-jobs 30 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri4b_ali_si284 exp/tri4b_denlats_si284 exp/tri4b exp/tri4b_mmi
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_etc.sh exp/tri4b/graph_tgpr data/test_dev93 exp/tri4b_mmi/decode_tgpr_dev93 exp/tri4b/decode_tgpr_dev93
steps/train_lda_etc_mmi.sh --boost 0.1 --num-jobs 30 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri4b_ali_si284 exp/tri4b_denlats_si284 exp/tri4b exp/tri4b_mmi_b0.1
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_etc.sh exp/tri4b/graph_tgpr data/test_dev93 exp/tri4b_mmi_b0.1/decode_tgpr_dev93 exp/tri4b/decode_tgpr_dev93

# Train UBM, for SGMM system on top of LDA+MLLT.
steps/train_ubm_lda_etc.sh --num-jobs 10 --cmd "$train_cmd" \
  400 data/train_si84 data/lang exp/tri2b_ali_si84 exp/ubm3c
steps/train_sgmm_lda_etc.sh --num-jobs 10 --cmd "$train_cmd" \
   3500 10000 41 40 data/train_si84 data/lang exp/tri2b_ali_si84 exp/ubm3c/final.ubm exp/sgmm3c
scripts/mkgraph.sh data/lang_test_tgpr exp/sgmm3c exp/sgmm3c/graph_tgpr
scripts/decode.sh --cmd "$decode_cmd" steps/decode_sgmm_lda_etc.sh exp/sgmm3c/graph_tgpr data/test_dev93 exp/sgmm3c/decode_tgpr_dev93
 

# Train SGMM system on top of LDA+MLLT+SAT.
steps/align_lda_mllt_sat.sh --num-jobs 10 --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri3b exp/tri3b_ali_si84
steps/train_ubm_lda_etc.sh --num-jobs 10 --cmd "$train_cmd" \
  400 data/train_si84 data/lang exp/tri3b_ali_si84 exp/ubm4b
steps/train_sgmm_lda_etc.sh  --num-jobs 10 --cmd "$train_cmd" \
  3500 10000 41 40 data/train_si84 data/lang exp/tri3b_ali_si84 exp/ubm4b/final.ubm exp/sgmm4b
scripts/mkgraph.sh data/lang_test_tgpr exp/sgmm4b exp/sgmm4b/graph_tgpr
scripts/decode.sh --cmd "$decode_cmd" steps/decode_sgmm_lda_etc.sh  \
    exp/sgmm4b/graph_tgpr data/test_dev93 exp/sgmm4b/decode_tgpr_dev93 exp/tri3b/decode_tgpr_dev93
scripts/decode.sh --cmd "$decode_cmd" steps/decode_sgmm_lda_etc.sh  \
    exp/sgmm4b/graph_tgpr data/test_eval92 exp/sgmm4b/decode_tgpr_eval92 exp/tri3b/decode_tgpr_eval92


# Align 3b system with si284 data and num-jobs = 20; we'll train an LDA+MLLT+SAT system on si284 from this.
# This is 4c.  c.f. 4b which is "quick" training.

steps/align_lda_mllt_sat.sh --num-jobs 20 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri3b exp/tri3b_ali_si284_20
steps/train_lda_mllt_sat.sh --num-jobs 20 --cmd "$train_cmd" \
  4200 40000 data/train_si284 data/lang exp/tri3b_ali_si284_20 exp/tri4c
scripts/mkgraph.sh data/lang_test_tgpr exp/tri4c exp/tri4c/graph_tgpr
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh exp/tri4c/graph_tgpr \
  data/test_dev93 exp/tri4c/decode_tgpr_dev93

# Train SGMM on top of LDA+MLLT+SAT, on all SI-284 data.  C.f. 4b which was
# just on SI-84.
steps/train_ubm_lda_etc.sh --num-jobs 20 --cmd "$train_cmd" \
  600 data/train_si284 data/lang exp/tri3b_ali_si284_20 exp/ubm4c
steps/train_sgmm_lda_etc.sh  --num-jobs 20 --cmd "$train_cmd" \
  5500 25000 50 40 data/train_si284 data/lang exp/tri3b_ali_si284_20 exp/ubm4c/final.ubm exp/sgmm4c
scripts/mkgraph.sh data/lang_test_tgpr exp/sgmm4c exp/sgmm4c/graph_tgpr
scripts/decode.sh --cmd "$decode_cmd" steps/decode_sgmm_lda_etc.sh  \
    exp/sgmm4c/graph_tgpr data/test_dev93 exp/sgmm4c/decode_tgpr_dev93 exp/tri3b/decode_tgpr_dev93
scripts/lmrescore.sh --cmd "$decode_cmd" data/lang_test_tgpr data/lang_test_tg \
  data/test_dev93 exp/sgmm4c/decode_tgpr_dev93 exp/sgmm4c/decode_tgpr_dev93_tg

# decode the above with nov'92 too
scripts/decode.sh --cmd "$decode_cmd" steps/decode_sgmm_lda_etc.sh  \
    exp/sgmm4c/graph_tgpr data/test_eval92 exp/sgmm4c/decode_tgpr_eval92 exp/tri3b/decode_tgpr_eval92
scripts/lmrescore.sh --cmd "$decode_cmd" data/lang_test_tgpr data/lang_test_tg \
  data/test_eval92 exp/sgmm4c/decode_tgpr_eval92 exp/sgmm4c/decode_tgpr_eval92_tg




# Getting results:
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | scripts/best_wer.sh; done
