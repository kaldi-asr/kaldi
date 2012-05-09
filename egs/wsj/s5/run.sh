#!/bin/bash

# WARNING: this is under construction.  Should stabilize by the end of May 2012.
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

#if false; then #TEMP

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

# local/wsj_data_prep.sh /mnt/matylda2/data/WSJ?/??-{?,??}.? || exit 1;

local/wsj_data_prep.sh  /export/corpora5/LDC/LDC{93S6,94S13}B/??-{?,??}.? || exit 1;

local/wsj_prepare_dict.sh || exit 1;

utils/prepare_lang.sh data/local/dict "<SPOKEN_NOISE>" data/local/lang data/lang || exit 1;

local/wsj_format_data.sh || exit 1;

# # We suggest to run the next three commands in the background,
# # as they are not a precondition for the system building and
# # most of the tests: these commands build a dictionary
# # containing many of the OOVs in the WSJ LM training data,
# # and an LM trained directly on that data (i.e. not just
# # copying the arpa files from the disks from LDC).
# (
#  on CSLP: local/wsj_extend_dict.sh /export/corpora5/LDC/LDC94S13B/13-32.1/ && \
#  local/wsj_extend_dict.sh /mnt/matylda2/data/WSJ1/13-32.1  && \
#  utils/prepare_lang.sh data/local/dict_larger "<SPOKEN_NOISE>" data/local/lang_larger data/lang_bd && \
#  local/wsj_train_lms.sh && \
#  local/wsj_format_local_lms.sh && 
#   (  local/wsj_train_rnnlms.sh --cmd "$train_cmd -l mem_free=10G" data/local/rnnlm.h30.voc10k &
#       sleep 20; # wait till tools compiled.
#     local/wsj_train_rnnlms.sh --cmd "$train_cmd -l mem_free=12G" \
#      --hidden 100 --nwords 20000 --class 350 --direct 1500 data/local/rnnlm.h100.voc20k &
#     local/wsj_train_rnnlms.sh --cmd "$train_cmd -l mem_free=14G" \
#      --hidden 200 --nwords 30000 --class 350 --direct 1500 data/local/rnnlm.h200.voc30k &
#     local/wsj_train_rnnlms.sh --cmd "$train_cmd -l mem_free=16G" \
#      --hidden 300 --nwords 40000 --class 400 --direct 2000 data/local/rnnlm.h300.voc40k &
#   )

# ) &

# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc
for x in test_eval92 test_eval93 test_dev93 train_si284; do 
 steps/make_mfcc.sh --cmd "$train_cmd" --nj 20 \
   data/$x exp/make_mfcc/$x $mfccdir || exit 1;
 steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
done

utils/subset_data_dir.sh --first data/train_si284 7138 data/train_si84 || exit 1

# Now make subset with the shortest 2k utterances from si-84.
utils/subset_data_dir.sh --shortest data/train_si84 2000 data/train_si84_2kshort || exit 1;

# Now make subset with half of the data from si-84.
utils/subset_data_dir.sh data/train_si84 3500 data/train_si84_half || exit 1;

steps/train_mono.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
  data/train_si84_2kshort data/lang exp/mono0a || exit 1;

(
 utils/mkgraph.sh --mono data/lang_test_tgpr exp/mono0a exp/mono0a/graph_tgpr && \
 steps/decode_si.sh --nj 10 --cmd "$decode_cmd" \
      exp/mono0a/graph_tgpr data/test_dev93 exp/mono0a/decode_tgpr_dev93 && \
 steps/decode_si.sh --nj 8 --cmd "$decode_cmd" \
   exp/mono0a/graph_tgpr data/test_eval92 exp/mono0a/decode_tgpr_eval92 
) &

steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
   data/train_si84_half data/lang exp/mono0a exp/mono0a_ali || exit 1;

steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/train_si84_half data/lang exp/mono0a_ali exp/tri1 || exit 1;

wait; # or the mono mkgraph.sh might be writing 
# data/lang_test_tgpr/tmp/LG.fst which will cause this to fail.

utils/mkgraph.sh data/lang_test_tgpr exp/tri1 exp/tri1/graph_tgpr || exit 1;


steps/decode_si.sh --nj 10 --cmd "$decode_cmd" \
  exp/tri1/graph_tgpr data/test_dev93 exp/tri1/decode_tgpr_dev93 || exit 1;
steps/decode_si.sh --nj 8 --cmd "$decode_cmd" \
  exp/tri1/graph_tgpr data/test_eval92 exp/tri1/decode_tgpr_eval92 || exit 1;


# test various modes of LM rescoring (4 is the default one).
# This is just confirming they're equivalent.
for mode in 1 2 3 4; do
steps/lmrescore.sh --mode $mode --cmd "$decode_cmd" data/lang_test_{tgpr,tg} \
  data/test_dev93 exp/tri1/decode_tgpr_dev93 exp/tri1/decode_tgpr_dev93_tg$mode  || exit 1;
done

sil_label=`grep '!SIL' data/lang_test_tgpr/words.txt | awk '{print 2}'`
steps/word_align_lattices.sh --cmd "$train_cmd" --silence-label $sil_label \
  data/lang_test_tgpr exp/tri1/decode_tgpr_dev93 exp/tri1/decode_tgpr_dev93_aligned || exit 1;


# Align tri1 system with si84 data.
steps/align_si.sh --nj 10 --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri1 exp/tri1_ali_si84 || exit 1;


# Train tri2a, which is deltas + delta-deltas, on si84 data.
steps/train_deltas.sh --cmd "$train_cmd" \
  2500 15000 data/train_si84 data/lang exp/tri1_ali_si84 exp/tri2a || exit 1;

utils/mkgraph.sh data/lang_test_tgpr exp/tri2a exp/tri2a/graph_tgpr || exit 1;

steps/decode_si.sh --nj 10 --cmd "$decode_cmd" \
  exp/tri2a/graph_tgpr data/test_dev93 exp/tri2a/decode_tgpr_dev93 || exit 1;

steps/train_lda_mllt.sh --cmd "$train_cmd" \
   2500 15000 data/train_si84 data/lang exp/tri1_ali_si84 exp/tri2b || exit 1;

utils/mkgraph.sh data/lang_test_tgpr exp/tri2b exp/tri2b/graph_tgpr || exit 1;
steps/decode_si.sh --nj 10 --cmd "$decode_cmd" \
  exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b/decode_tgpr_dev93 || exit 1;
steps/decode_si.sh --nj 8 --cmd "$decode_cmd" \
  exp/tri2b/graph_tgpr data/test_eval92 exp/tri2b/decode_tgpr_eval92 || exit 1;


# Now, with dev93, compare lattice rescoring with biglm decoding,
# going from tgpr to tg.  Note: results are not the same, even though they should
# be, and I believe this is due to the beams not being wide enough.  The pruning
# seems to be a bit too narrow in the current scripts (got at least 0.7% absolute
# improvement from loosening beams from their current values).

steps/decode_si_biglm.sh --nj 10 --cmd "$decode_cmd" \
  exp/tri2b/graph_tgpr data/lang_test_{tgpr,tg}/G.fst \
  data/test_dev93 exp/tri2b/decode_tgpr_dev93_tg_biglm

# baseline via LM rescoring of lattices.
steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_tgpr/ data/lang_test_tg/ \
  data/test_dev93 exp/tri2b/decode_tgpr_dev93 exp/tri2b/decode_tgpr_dev93_tg || exit 1;

# Trying Minimum Bayes Risk decoding (like Confusion Network decoding):
mkdir exp/tri2b/decode_tgpr_dev93_tg_mbr 
cp exp/tri2b/decode_tgpr_dev93_tg/lat.*.gz exp/tri2b/decode_tgpr_dev93_tg_mbr 
local/score_mbr.sh --cmd "$decode_cmd" \
 data/test_dev93/ data/lang_test_tgpr/ exp/tri2b/decode_tgpr_dev93_tg_mbr

steps/decode_si_fromlats.sh --cmd "$decode_cmd" \
  data/test_dev93 data/lang_test_tgpr exp/tri2b/decode_tgpr_dev93 \
  exp/tri2a/decode_tgpr_dev93_fromlats || exit 1;



# Align tri2b system with si84 data.
steps/align_si.sh  --nj 10 --cmd "$train_cmd" \
  --use-graphs true data/train_si84 data/lang exp/tri2b exp/tri2b_ali_si84  || exit 1;


# Train and test MMI (and boosted MMI) on tri2b system.
steps/make_denlats.sh --sub-split 20 --nj 10 --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri2b exp/tri2b_denlats_si84 || exit 1;

steps/train_mmi.sh --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri2b_ali_si84 \
  exp/tri2b_denlats_si84 exp/tri2b_mmi  || exit 1;
 (
  steps/make_denlats.sh --sub-split 20 --nj 10 --cmd "$train_cmd" \
   data/train_si84 data/lang exp/tri2b_mmi exp/tri2b_denlats_si84_2 || exit 1;
  steps/align_si.sh  --nj 10 --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri2b_mmi exp/tri2b_ali_si84_2  || exit 1;
  steps/train_mmi.sh --cmd "$train_cmd" \
    data/train_si84 data/lang exp/tri2b_ali_si84_2 \
    exp/tri2b_denlats_si84_2 exp/tri2b_mmi_2  || exit 1;
  steps/decode_si.sh --nj 10 --cmd "$decode_cmd" \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_mmi_2/decode_tgpr_dev93 || exit 1;
  steps/decode_si.sh --nj 8 --cmd "$decode_cmd" \
    exp/tri2b/graph_tgpr data/test_eval92 exp/tri2b_mmi_2/decode_tgpr_eval92 || exit 1;
 )


steps/decode_si.sh --nj 10 --cmd "$decode_cmd" \
  exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_mmi/decode_tgpr_dev93 || exit 1;
steps/decode_si.sh --nj 8 --cmd "$decode_cmd" \
   exp/tri2b/graph_tgpr data/test_eval92 exp/tri2b_mmi/decode_tgpr_eval92 || exit 1;

steps/train_mmi.sh --cmd "$train_cmd" --boost 0.1 \
  data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri2b_denlats_si84 \
  exp/tri2b_mmi_b0.1  || exit 1;

steps/decode_si.sh --nj 10 --cmd "$decode_cmd" \
   exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_mmi_b0.1/decode_tgpr_dev93 || exit 1;
steps/decode_si.sh --nj 8 --cmd "$decode_cmd" \
   exp/tri2b/graph_tgpr data/test_eval92 exp/tri2b_mmi_b0.1/decode_tgpr_eval92 || exit 1;

 # Test iters 2 and 3
 steps/decode_si.sh --iter 2 --nj 10 --cmd "$decode_cmd" \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_mmi_b0.1/decode_tgpr_dev93.it2
 steps/decode_si.sh --iter 3 --nj 10 --cmd "$decode_cmd" \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_mmi_b0.1/decode_tgpr_dev93.it3

 # The next 3 commands train and test fMMI+MMI (on top of LDA+MLLT).
 steps/train_diag_ubm.sh --silence-weight 0.5 --nj 10 --cmd "$train_cmd" \
   400 data/train_si84 data/lang exp/tri2b_ali_si84 exp/dubm2b

 steps/train_mmi_fmmi.sh --boost 0.1 --cmd "$train_cmd" \
   data/train_si84 data/lang exp/tri2b_ali_si84 exp/dubm2b exp/tri2b_denlats_si84 \
   exp/tri2b_fmmi_b0.1

 steps/decode_fmmi.sh --nj 10 --cmd "$decode_cmd" --iter 4 \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_fmmi_b0.1/decode_tgpr_dev93_it4 &
 steps/decode_fmmi.sh --nj 10 --cmd "$decode_cmd" --iter 5 \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_fmmi_b0.1/decode_tgpr_dev93_it5 &
 steps/decode_fmmi.sh --nj 10 --cmd "$decode_cmd" --iter 6 \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_fmmi_b0.1/decode_tgpr_dev93_it6 &
 steps/decode_fmmi.sh --nj 10 --cmd "$decode_cmd" --iter 7 \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_fmmi_b0.1/decode_tgpr_dev93_it7 &
 steps/decode_fmmi.sh --nj 10 --cmd "$decode_cmd" --iter 8 \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_fmmi_b0.1/decode_tgpr_dev93_it8 &

 steps/train_mmi_fmmi.sh --learning-rate 0.005 --boost 0.1 --cmd "$train_cmd" \
   data/train_si84 data/lang exp/tri2b_ali_si84 exp/dubm2b exp/tri2b_denlats_si84 \
   exp/tri2b_fmmi_b0.1_lr0.005 || exit 1;
 
 steps/decode_fmmi.sh --nj 10 --cmd "$decode_cmd" --iter 4 \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_fmmi_b0.1_lr0.005/decode_tgpr_dev93_it4 &
 steps/decode_fmmi.sh --nj 10 --cmd "$decode_cmd" --iter 5 \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_fmmi_b0.1_lr0.005/decode_tgpr_dev93_it5 &
 steps/decode_fmmi.sh --nj 10 --cmd "$decode_cmd" --iter 6 \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_fmmi_b0.1_lr0.005/decode_tgpr_dev93_it6 &
 steps/decode_fmmi.sh --nj 10 --cmd "$decode_cmd" --iter 7 \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_fmmi_b0.1_lr0.005/decode_tgpr_dev93_it7 &
 steps/decode_fmmi.sh --nj 10 --cmd "$decode_cmd" --iter 8 \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_fmmi_b0.1_lr0.005/decode_tgpr_dev93_it8 &

 steps/train_mmi_fmmi_indirect.sh --boost 0.1 --cmd "$train_cmd" \
   data/train_si84 data/lang exp/tri2b_ali_si84 exp/dubm2b exp/tri2b_denlats_si84 \
   exp/tri2b_fmmi_indirect_b0.1
 steps/decode_fmmi.sh --nj 10 --cmd "$decode_cmd" --iter 4 \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_fmmi_indirect_b0.1/decode_tgpr_dev93_it4 &
 steps/decode_fmmi.sh --nj 10 --cmd "$decode_cmd" --iter 5 \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_fmmi_indirect_b0.1/decode_tgpr_dev93_it5 &
 steps/decode_fmmi.sh --nj 10 --cmd "$decode_cmd" --iter 6 \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_fmmi_indirect_b0.1/decode_tgpr_dev93_it6 &
 steps/decode_fmmi.sh --nj 10 --cmd "$decode_cmd" --iter 7 \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_fmmi_indirect_b0.1/decode_tgpr_dev93_it7 &
 steps/decode_fmmi.sh --nj 10 --cmd "$decode_cmd" --iter 8 \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_fmmi_indirect_b0.1/decode_tgpr_dev93_it8 &

 steps/train_mmi_fmmi.sh --learning-rate 0.005 --schedule "fmmi fmmi fmmi fmmi mmi mmi mmi mmi" \
    --boost 0.1 --cmd "$train_cmd" \
   data/train_si84 data/lang exp/tri2b_ali_si84 exp/dubm2b exp/tri2b_denlats_si84 \
   exp/tri2b_fmmi_first_b0.1 || exit 1;
 
 steps/decode_fmmi.sh --nj 10 --cmd "$decode_cmd" --iter 4 \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_fmmi_first_b0.1/decode_tgpr_dev93_it4 &
 steps/decode_fmmi.sh --nj 10 --cmd "$decode_cmd" --iter 5 \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_fmmi_first_b0.1/decode_tgpr_dev93_it5 &
 steps/decode_fmmi.sh --nj 10 --cmd "$decode_cmd" --iter 6 \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_fmmi_first_b0.1/decode_tgpr_dev93_it6 &
 steps/decode_fmmi.sh --nj 10 --cmd "$decode_cmd" --iter 7 \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_fmmi_first_b0.1/decode_tgpr_dev93_it7 &
 steps/decode_fmmi.sh --nj 10 --cmd "$decode_cmd" --iter 8 \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_fmmi_first_b0.1/decode_tgpr_dev93_it8 &

#fi #TEMP

# From 2b system, train 3b which is LDA + MLLT + SAT.
steps/train_sat.sh  --cmd "$train_cmd" \
  2500 15000 data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri3b || exit 1;
utils/mkgraph.sh data/lang_test_tgpr exp/tri3b exp/tri3b/graph_tgpr || exit 1;
steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
  exp/tri3b/graph_tgpr data/test_dev93 exp/tri3b/decode_tgpr_dev93 || exit 1;
steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
  exp/tri3b/graph_tgpr data/test_eval92 exp/tri3b/decode_tgpr_eval92 || exit 1;


 # steps/decode_fmllr_thresh.sh --nj 10 --cmd "$decode_cmd" \
 #   exp/tri3b/graph_tgpr data/test_dev93 exp/tri3b/decode_tgpr_dev93_thresh || exit 1;
 # steps/decode_fmllr_thresh.sh --nj 8 --cmd "$decode_cmd" \
 #   exp/tri3b/graph_tgpr data/test_eval92 exp/tri3b/decode_tgpr_eval92_thresh || exit 1;

 # steps/decode_fmllr_thresh.sh --threshold 0.99 --nj 10 --cmd "$decode_cmd" \
 #   exp/tri3b/graph_tgpr data/test_dev93 exp/tri3b/decode_tgpr_dev93_thresh_2 || exit 1;
 # steps/decode_fmllr_thresh.sh --threshold 0.99 --nj 8 --cmd "$decode_cmd" \
 #   exp/tri3b/graph_tgpr data/test_eval92 exp/tri3b/decode_tgpr_eval92_thresh_2 || exit 1;


steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_tgpr data/lang_test_tg \
  data/test_dev93 exp/tri3b/decode_tgpr_dev93 exp/tri3b/decode_tgpr_dev93_tg || exit 1;
steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_tgpr data/lang_test_tg \
  data/test_eval92 exp/tri3b/decode_tgpr_eval92 exp/tri3b/decode_tgpr_eval92_tg || exit 1;


# Trying the larger dictionary ("big-dict"/bd) + locally produced LM.
utils/mkgraph.sh data/lang_test_bd_tgpr exp/tri3b exp/tri3b/graph_bd_tgpr || exit 1;

steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 8 \
  exp/tri3b/graph_bd_tgpr data/test_eval92 exp/tri3b/decode_bd_tgpr_eval92 || exit 1;
steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 10 \
  exp/tri3b/graph_bd_tgpr data/test_dev93 exp/tri3b/decode_bd_tgpr_dev93 || exit 1;

steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_bd_tgpr data/lang_test_bd_fg \
  data/test_eval92 exp/tri3b/decode_bd_tgpr_eval92 exp/tri3b/decode_bd_tgpr_eval92_fg \
   || exit 1;
steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_bd_tgpr data/lang_test_bd_tg \
  data/test_eval92 exp/tri3b/decode_bd_tgpr_eval92 exp/tri3b/decode_bd_tgpr_eval92_tg \
  || exit 1;

 # This step interpolates a small RNNLM (with weight 0.25) with the 4-gram LM.
 steps/rnnlmrescore.sh \
  --inv-acwt 17 \
  0.25 data/lang_test_bd_fg data/local/rnnlm.h30.voc10k data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm30_0.25  \
  || exit 1;


# The following two steps, which are a kind of side-branch, try mixing up
( # from the 3b system.  This is to demonstrate that script.
 steps/mixup.sh --cmd "$train_cmd" \
   20000 data/train_si84 data/lang exp/tri3b exp/tri3b_20k || exit 1;
 steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 10 \
   exp/tri3b/graph_tgpr data/test_dev93 exp/tri3b_20k/decode_tgpr_dev93  || exit 1;
)


# From 3b system, align all si284 data.
steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri3b exp/tri3b_ali_si284 || exit 1;


# From 3b system, train another SAT system with all the si284 data.
# Use the letter tri4a, as tri4b was used in s3/ for a "quick-retrained" system.
steps/train_sat.sh  --nj 20 --cmd "$train_cmd" \
  4200 40000 data/train_si284 data/lang exp/tri3b_ali_si284 exp/tri4a || exit 1;

utils/mkgraph.sh data/lang_test_tgpr exp/tri4a exp/tri4a/graph_tgpr || exit 1;
steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
  exp/tri4a/graph_tgpr data/test_dev93 exp/tri4a/decode_tgpr_dev93 || exit 1;
steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
  exp/tri4a/graph_tgpr data/test_eval92 exp/tri4a/decode_tgpr_eval92 || exit 1;

steps/train_quick.sh --cmd "$train_cmd" \
   4200 40000 data/train_si284 data/lang exp/tri3b_ali_si284 exp/tri4b || exit 1;

utils/mkgraph.sh data/lang_test_tgpr exp/tri4b exp/tri4b/graph_tgpr || exit 1;
steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
  exp/tri4b/graph_tgpr data/test_dev93 exp/tri4b/decode_tgpr_dev93 || exit 1;
steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
  exp/tri4b/graph_tgpr data/test_eval92 exp/tri4b/decode_tgpr_eval92 || exit 1;


# Train and test MMI, and boosted MMI, on tri4b (LDA+MLLT+SAT on
# all the data).  Use 30 jobs.
steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri4b exp/tri4b_ali_si284 || exit 1;

steps/make_denlats.sh --nj 30 --sub-split 30 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri4b exp/tri4b_denlats_si284 || exit 1;

steps/train_mmi.sh --cmd "$train_cmd" --boost 0.1 \
  data/train_si284 data/lang exp/tri4b_ali_si284 exp/tri4b_denlats_si284 \
  exp/tri4b_mmi_b0.1  || exit 1;

## I AM HERE

steps/train_lda_etc_mmi.sh --nj 40 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri4b_ali_si284 exp/tri4b_denlats_si284 \
  exp/tri4b exp/tri4b_mmi || exit 1;
utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_etc.sh exp/tri4b/graph_tgpr \
  data/test_dev93 exp/tri4b_mmi/decode_tgpr_dev93 exp/tri4b/decode_tgpr_dev93 \
   || exit 1;
steps/train_lda_etc_mmi.sh --boost 0.1 --nj 40 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri4b_ali_si284 exp/tri4b_denlats_si284 \
  exp/tri4b exp/tri4b_mmi_b0.1 || exit 1;
utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_etc.sh exp/tri4b/graph_tgpr \
 data/test_dev93 exp/tri4b_mmi_b0.1/decode_tgpr_dev93 exp/tri4b/decode_tgpr_dev93 \
   || exit 1;
 utils/decode.sh --opts "--beam 15" --cmd "$decode_cmd" steps/decode_lda_etc.sh exp/tri4b/graph_tgpr data/test_dev93 exp/tri4b_mmi_b0.1/decode_tgpr_dev93_b15 exp/tri4b/decode_tgpr_dev93
utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_etc.sh exp/tri4b/graph_tgpr data/test_eval92 exp/tri4b_mmi_b0.1/decode_tgpr_eval92 exp/tri4b/decode_tgpr_eval92

 # Train fMMI+MMI system on top of 4b.
 steps/train_dubm_lda_etc.sh --silence-weight 0.5 \
   --nj 40 --cmd "$train_cmd" 600 data/train_si284 \
   data/lang exp/tri4b_ali_si284 exp/dubm4b
 steps/train_lda_etc_mmi_fmmi.sh \
   --nj 40 --boost 0.1 --cmd "$train_cmd" \
   data/train_si284 data/lang exp/tri4b_ali_si284 exp/dubm4b exp/tri4b_denlats_si284 \
   exp/tri4b exp/tri4b_fmmi_b0.1 
 utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_etc_fmpe.sh \
   exp/tri4b/graph_tgpr data/test_eval92 exp/tri4b_fmmi_b0.1/decode_tgpr_eval92 \
   exp/tri4b/decode_tgpr_eval92
 utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_etc_fmpe.sh \
   exp/tri4b/graph_tgpr data/test_dev93 exp/tri4b_fmmi_b0.1/decode_tgpr_dev93 \
   exp/tri4b/decode_tgpr_dev93



# Train UBM, for SGMM system on top of LDA+MLLT.
steps/train_ubm_lda_etc.sh --nj 10 --cmd "$train_cmd" \
  400 data/train_si84 data/lang exp/tri2b_ali_si84 exp/ubm3c || exit 1;
steps/train_sgmm_lda_etc.sh --nj 10 --cmd "$train_cmd" \
   3500 10000 41 40 data/train_si84 data/lang exp/tri2b_ali_si84 \
   exp/ubm3c/final.ubm exp/sgmm3c || exit 1;
utils/mkgraph.sh data/lang_test_tgpr exp/sgmm3c exp/sgmm3c/graph_tgpr || exit 1;
utils/decode.sh --cmd "$decode_cmd" steps/decode_sgmm_lda_etc.sh \
  exp/sgmm3c/graph_tgpr data/test_dev93 exp/sgmm3c/decode_tgpr_dev93 || exit 1;



# Decode using 3 Gaussians (not 15) for gselect in 1st pass, for fast decoding.
utils/decode.sh --opts "--first-pass-gselect 3" --cmd "$decode_cmd" \
  steps/decode_sgmm_lda_etc.sh exp/sgmm3c/graph_tgpr data/test_dev93 \
  exp/sgmm3c/decode_tgpr_dev93_gs3 || exit 1;

# Decoding via lattice rescoring of lats from regular model. [ a bit worse].
utils/decode.sh --cmd "$decode_cmd" steps/decode_sgmm_lda_etc_fromlats.sh \
  data/lang_test_tgpr data/test_dev93 exp/sgmm3c/decode_tgpr_dev93_fromlats \
  exp/tri2b/decode_tgpr_dev93 || exit 1;

# Train SGMM system on top of LDA+MLLT+SAT.
steps/align_lda_mllt_sat.sh --nj 10 --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri3b exp/tri3b_ali_si84 || exit 1;
steps/train_ubm_lda_etc.sh --nj 10 --cmd "$train_cmd" \
  400 data/train_si84 data/lang exp/tri3b_ali_si84 exp/ubm4b || exit 1;
steps/train_sgmm_lda_etc.sh  --nj 10 --cmd "$train_cmd" \
  3500 10000 41 40 data/train_si84 data/lang exp/tri3b_ali_si84 \
 exp/ubm4b/final.ubm exp/sgmm4b || exit 1;
utils/mkgraph.sh data/lang_test_tgpr exp/sgmm4b exp/sgmm4b/graph_tgpr
utils/decode.sh --cmd "$decode_cmd" steps/decode_sgmm_lda_etc.sh  \
  exp/sgmm4b/graph_tgpr data/test_dev93 exp/sgmm4b/decode_tgpr_dev93 \
 exp/tri3b/decode_tgpr_dev93 || exit 1;
utils/decode.sh --cmd "$decode_cmd" steps/decode_sgmm_lda_etc.sh  \
  exp/sgmm4b/graph_tgpr data/test_eval92 exp/sgmm4b/decode_tgpr_eval92 \
  exp/tri3b/decode_tgpr_eval92 || exit 1;

 # Trying further mixing-up of this system [increase #substates]:
  steps/mixup_sgmm_lda_etc.sh --nj 10 --cmd "$train_cmd" \
     12500 data/train_si84 exp/sgmm4b exp/tri3b_ali_si84 exp/sgmm4b_12500 || exit 1;
  utils/decode.sh --cmd "$decode_cmd" steps/decode_sgmm_lda_etc.sh  \
    exp/sgmm4b/graph_tgpr data/test_eval92 exp/sgmm4b_12500/decode_tgpr_eval92 exp/tri3b/decode_tgpr_eval92 || exit 1;
 # note: taking it up to 150k made it worse again [8.63->8.56->8.72 ... this was before some
 # decoding-script changes so these results not the same as in RESULTS file.]
  # increasing phone dim but not #substates..
  steps/mixup_sgmm_lda_etc.sh --nj 10 --cmd "$train_cmd" --increase-phone-dim 50 \
    10000 data/train_si84 exp/sgmm4b exp/tri3b_ali_si84 exp/sgmm4b_50 || exit 1;
  utils/decode.sh --cmd "$decode_cmd" steps/decode_sgmm_lda_etc.sh  \
    exp/sgmm4b/graph_tgpr data/test_eval92 exp/sgmm4b_50/decode_tgpr_eval92 \
   exp/tri3b/decode_tgpr_eval92 || exit 1;

# Align 3b system with si284 data and num-jobs = 20; we'll train an LDA+MLLT+SAT system on si284 from this.
# This is 4c.  c.f. 4b which is "quick" training.

steps/align_lda_mllt_sat.sh --nj 20 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri3b exp/tri3b_ali_si284_20 || exit 1;
steps/train_lda_mllt_sat.sh --nj 20 --cmd "$train_cmd" \
  4200 40000 data/train_si284 data/lang exp/tri3b_ali_si284_20 exp/tri4c || exit 1;
utils/mkgraph.sh data/lang_test_tgpr exp/tri4c exp/tri4c/graph_tgpr || exit 1;
utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh \
  exp/tri4c/graph_tgpr data/test_dev93 exp/tri4c/decode_tgpr_dev93 || exit 1;


( # Try mixing up the tri4c system further.
 steps/mixup_lda_etc.sh --nj 20 --cmd "$train_cmd" \
  50000 data/train_si284 exp/tri4c exp/tri3b_ali_si284_20 exp/tri4c_50k || exit 1;
 utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh \
  exp/tri4c/graph_tgpr data/test_dev93 exp/tri4c_50k/decode_tgpr_dev93 || exit 1;

 steps/mixup_lda_etc.sh --nj 20 --cmd "$train_cmd" \
  75000 data/train_si284 exp/tri4c_50k exp/tri3b_ali_si284_20 exp/tri4c_75k || exit 1;
 utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh \
   exp/tri4c/graph_tgpr data/test_dev93 exp/tri4c_75k/decode_tgpr_dev93 || exit 1;

 steps/mixup_lda_etc.sh --nj 20 --cmd "$train_cmd" \
  100000 data/train_si284 exp/tri4c_75k exp/tri3b_ali_si284_20 exp/tri4c_100k || exit 1;
 utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh \
  exp/tri4c/graph_tgpr data/test_dev93 exp/tri4c_100k/decode_tgpr_dev93 || exit 1;
)


# Train SGMM on top of LDA+MLLT+SAT, on all SI-284 data.  C.f. 4b which was
# just on SI-84.
steps/train_ubm_lda_etc.sh --nj 20 --cmd "$train_cmd" \
  600 data/train_si284 data/lang exp/tri3b_ali_si284_20 exp/ubm4c || exit 1;
steps/train_sgmm_lda_etc.sh  --nj 20 --cmd "$train_cmd" \
  5500 25000 50 40 data/train_si284 data/lang exp/tri3b_ali_si284_20 \
  exp/ubm4c/final.ubm exp/sgmm4c || exit 1;
utils/mkgraph.sh data/lang_test_tgpr exp/sgmm4c exp/sgmm4c/graph_tgpr || exit 1;
utils/decode.sh --cmd "$decode_cmd" steps/decode_sgmm_lda_etc.sh  \
  exp/sgmm4c/graph_tgpr data/test_dev93 exp/sgmm4c/decode_tgpr_dev93 \
  exp/tri3b/decode_tgpr_dev93  || exit 1;
utils/lmrescore.sh --cmd "$decode_cmd" data/lang_test_tgpr data/lang_test_tg \
  data/test_dev93 exp/sgmm4c/decode_tgpr_dev93 exp/sgmm4c/decode_tgpr_dev93_tg \
  || exit 1;

# decode sgmm4c with nov'92 too
utils/decode.sh --cmd "$decode_cmd" steps/decode_sgmm_lda_etc.sh  \
    exp/sgmm4c/graph_tgpr data/test_eval92 exp/sgmm4c/decode_tgpr_eval92 \
  exp/tri3b/decode_tgpr_eval92 || exit 1;
utils/lmrescore.sh --cmd "$decode_cmd" data/lang_test_tgpr data/lang_test_tg \
  data/test_eval92 exp/sgmm4c/decode_tgpr_eval92 exp/sgmm4c/decode_tgpr_eval92_tg \
  || exit 1;

# Decode sgmm4c with the "big-dict" decoding graph.
utils/mkgraph.sh data/lang_test_bd_tgpr exp/sgmm4c exp/sgmm4c/graph_bd_tgpr || exit 1;
utils/decode.sh --cmd "$decode_cmd" steps/decode_sgmm_lda_etc.sh \
  exp/sgmm4c/graph_bd_tgpr data/test_eval92 exp/sgmm4c/decode_bd_tgpr_eval92 \
 exp/tri3b/decode_tgpr_eval92

# The same with Gaussian selection pruned down, for speed.
utils/decode.sh --opts "--first-pass-gselect 3" --cmd "$decode_cmd" \
   steps/decode_sgmm_lda_etc.sh exp/sgmm4c/graph_bd_tgpr \
  data/test_eval92 exp/sgmm4c/decode_bd_tgpr_eval92_gs3 exp/tri3b/decode_tgpr_eval92 \
  || exit 1;
utils/lmrescore.sh --cmd "$decode_cmd" data/lang_test_bd_tgpr data/lang_test_bd_fg \
  data/test_eval92 exp/sgmm4c/decode_bd_tgpr_eval92 exp/sgmm4c/decode_bd_tgpr_eval92_fg \
  || exit 1;
utils/lmrescore.sh --cmd "$decode_cmd" data/lang_test_bd_tgpr data/lang_test_bd_tg \
 data/test_eval92 exp/sgmm4c/decode_bd_tgpr_eval92 exp/sgmm4c/decode_bd_tgpr_eval92_tg \
  || exit 1;

# Decode sgmm4c with the "big-dict" decoding graph; here, we avoid doing a full
# re-decoding but limit ourselves from the lattices from the "big-dict" decoding
# of exp/tri3b.  Note that these results are not quite comparable to those
# above because we use the better, "big-dict" decoding of tri3b.  We go direct to
# 4-gram.
utils/decode.sh --cmd "$decode_cmd" steps/decode_sgmm_lda_etc_fromlats.sh \
 data/lang_test_bd_fg data/test_eval92 exp/sgmm4c/decode_bd_fg_eval92_fromlats \
  exp/tri3b/decode_bd_tgpr_eval92 || exit 1;


# Train quinphone SGMM system.  Note: this is not substantially
# better than the triphone one... it seems quinphone does not really
# help. Also, the likelihoods (after training) are barely improved.
steps/train_sgmm_lda_etc.sh  --nj 20 --cmd "$train_cmd" \
  --context-opts "--context-width=5 --central-position=2" \
  5500 25000 50 40 data/train_si284 data/lang exp/tri3b_ali_si284_20 \
  exp/ubm4c/final.ubm exp/sgmm4d || exit 1;

# This is a decoding via lattice rescoring... compare with the same for sgmm4c
utils/decode.sh --cmd "$decode_cmd" steps/decode_sgmm_lda_etc_fromlats.sh \
 data/lang_test_bd_fg data/test_eval92 exp/sgmm4d/decode_bd_fg_eval92_fromlats \
  exp/tri3b/decode_bd_tgpr_eval92 || exit 1;


 # This is the same, but from better lattices (from decoding sgmm4c).
 # Need to copy the transforms over from original decoding dir (tri3b) 
 # to the source dir for the lattices, as they're used by the decoding script.
 # This is a bit of a mess.
 cp exp/tri3b/decode_bd_tgpr_eval92/*.trans exp/sgmm4c/decode_bd_tgpr_eval92
 utils/decode.sh --cmd "$decode_cmd" steps/decode_sgmm_lda_etc_fromlats.sh \
 data/lang_test_bd_fg data/test_eval92 exp/sgmm4d/decode_bd_fg_eval92_fromlats2 \
   exp/sgmm4c/decode_bd_tgpr_eval92 || exit 1;


#Note: in preparation to a direct decode of sgmm4d, I ran the following command:
#utils/mkgraph.sh --quinphone data/lang_test_bd_tgpr exp/sgmm4d exp/sgmm4d/graph_bd_tgpr || exit 1;
# but I gave up after fstcomposecontext reached 22G and was still working.

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
