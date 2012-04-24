#!/bin/bash

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

local/wsj_data_prep.sh /mnt/matylda2/data/WSJ?/??-{?,??}.? || exit 1;

#local/wsj_data_prep.sh  /export/corpora5/LDC/LDC{93S6,94S13}B/??-{?,??}.? || exit 1;

local/wsj_prepare_dict.sh || exit 1;

utils/prepare_lang.sh data/local/dict data/local/lang data/lang || exit 1;

local/wsj_format_data.sh || exit 1;

# # We suggest to run the next three commands in the background,
# # as they are not a precondition for the system building and
# # most of the tests: these commands build a dictionary
# # containing many of the OOVs in the WSJ LM training data,
# # and an LM trained directly on that data (i.e. not just
# # copying the arpa files from the disks from LDC).
# (
# # on CSLP: local/wsj_extend_dict.sh /export/corpora5/LDC/LDC94S13B/13-32.1/ && \
#  local/wsj_extend_dict.sh /mnt/matylda2/data/WSJ1/13-32.1  && \
#  local/wsj_prepare_local_dict.sh && \
#  local/wsj_train_lms.sh && \
#  local/wsj_format_data_local.sh && \
#  local/wsj_train_rnnlms.sh
# ) &

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
          ## This relates to the queue.

# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc
for x in test_eval92 test_eval93 test_dev93 train_si284; do 
 steps/make_mfcc.sh --cmd "$train_cmd" --num-jobs 20 \
    data/$x exp/make_mfcc/$x $mfccdir || exit 1;
 steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
done

utils/subset_data_dir.sh --first data/train_si284 7138 data/train_si84 || exit 1

# Now make subset with the shortest 2k utterances from si-84.
utils/subset_data_dir.sh --shortest data/train_si84 2000 data/train_si84_2kshort || exit 1;

# Now make subset with half of the data from si-84.
utils/subset_data_dir.sh data/train_si84 3500 data/train_si84_half || exit 1;

steps/train_mono.sh --num-jobs 10 --cmd "$train_cmd" \
  data/train_si84_2kshort data/lang exp/mono0a || exit 1;

(
 utils/mkgraph.sh --mono data/lang_test_tgpr exp/mono0a exp/mono0a/graph_tgpr && \
 steps/decode_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
      exp/mono0a/graph_tgpr data/test_dev93 exp/mono0a/decode_tgpr_dev93 && \
 steps/decode_deltas.sh --num-jobs 8 --cmd "$train_cmd" \
   exp/mono0a/graph_tgpr data/test_eval92 exp/mono0a/decode_tgpr_eval92 
) &

steps/align_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
   data/train_si84_half data/lang exp/mono0a exp/mono0a_ali || exit 1;

steps/train_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
    2000 10000 data/train_si84_half data/lang exp/mono0a_ali exp/tri1 || exit 1;

wait; # or the mono mkgraph.sh might be writing 
# data/lang_test_tgpr/tmp/LG.fst which will cause this to fail.

utils/mkgraph.sh data/lang_test_tgpr exp/tri1 exp/tri1/graph_tgpr || exit 1;

steps/decode_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
  exp/tri1/graph_tgpr data/test_dev93 exp/tri1/decode_tgpr_dev93 || exit 1;
steps/decode_deltas.sh --num-jobs 8 --cmd "$train_cmd" \
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
steps/align_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri1 exp/tri1_ali_si84 || exit 1;


# Train tri2a, which is deltas + delta-deltas, on si84 data.
steps/train_deltas.sh  --num-jobs 10 --cmd "$train_cmd" \
  2500 15000 data/train_si84 data/lang exp/tri1_ali_si84 exp/tri2a || exit 1;

utils/mkgraph.sh data/lang_test_tgpr exp/tri2a exp/tri2a/graph_tgpr || exit 1;

steps/decode_deltas.sh --num-jobs 10 --cmd "$decode_cmd" \
  exp/tri2a/graph_tgpr data/test_dev93 exp/tri2a/decode_tgpr_dev93 || exit 1;

##################### I AM HERE ####################################


# Train tri2b, which is LDA+MLLT, on si84 data.
steps/train_lda_mllt.sh --num-jobs 10 --cmd "$train_cmd" \
   2500 15000 data/train_si84 data/lang exp/tri1_ali_si84 exp/tri2b || exit 1;
utils/mkgraph.sh data/lang_test_tgpr exp/tri2b exp/tri2b/graph_tgpr || exit 1;
utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt.sh exp/tri2b/graph_tgpr data/test_eval92 exp/tri2b/decode_tgpr_eval92 || exit 1;
utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt.sh exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b/decode_tgpr_dev93 || exit 1;

# Now, with dev93, compare lattice rescoring with biglm decoding,
# going from tgpr to tg.  Note: results are not the same, even though they should
# be, and I believe this is due to the beams not being wide enough.  The pruning
# seems to be a bit too narrow in the current scripts (got at least 0.7% absolute
# improvement from loosening beams from their current values).

# Note: if you are running this soon after it's created and not from scratch, you
# may have to rerun local/wsj_format_data.sh before the command below (a rmepsilon
# stage in there that's necessary).
utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_biglm.sh exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b/decode_tgpr_dev93_tg_biglm data/lang_test_tgpr/G.fst data/lang_test_tg/G.fst || exit 1;
# baseline via LM rescoring of lattices.
utils/lmrescore.sh --cmd "$decode_cmd" data/lang_test_tgpr/ data/lang_test_tg/ \
  data/test_dev93 exp/tri2b/decode_tgpr_dev93 exp/tri2b/decode_tgpr_dev93_tg || exit 1;

# Trying Minimum Bayes Risk decoding (like Confusion Network decoding):
mkdir exp/tri2b/decode_tgpr_dev93_tg_mbr 
cp exp/tri2b/decode_tgpr_dev93_tg/lat.*.gz exp/tri2b/decode_tgpr_dev93_tg_mbr 
utils/score_mbr.sh exp/tri2b/decode_tgpr_dev93_tg_mbr  data/lang_test_tgpr/words.txt data/test_dev93

# Demonstrate 'cross-tree' lattice rescoring where we create utterance-specific
# decoding graphs from one system's lattices and rescore with another system.
# Note: we could easily do this with the trigram LM, unpruned, but for comparability
# with the experiments above we do it with the pruned one.
utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_fromlats.sh \
  data/lang_test_tgpr data/test_dev93 exp/tri2b/decode_tgpr_dev93_fromlats \
  exp/tri2a/decode_tgpr_dev93 || exit 1;

# Align tri2b system with si84 data.
steps/align_lda_mllt.sh  --num-jobs 10 --cmd "$train_cmd" \
  --use-graphs data/train_si84 data/lang exp/tri2b exp/tri2b_ali_si84  || exit 1;

## HERE-- new stuff.
(
  cp exp/tri2b/*.fsts.gz exp/tri2b_ali_si84
  steps/train_raw_mllt_sat.sh --num-jobs 10 --cmd "$train_cmd" \
   2500 15000 data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri2b_raw || exit 1;

  steps/get_raw_transforms.sh  --num-jobs 10 --cmd "$train_cmd" \
    data/train_si84 data/lang exp/tri2b_raw exp/tri2b_ali_si84 exp/tri2b_raw_trans_si84

  # We need to be a bit careful about the #jobs, as the normal decoding
  # scripts use 4 if on local, else #spks.
  nj=`ls exp/tri2b/decode_tgpr_dev93/lat.*.gz | wc -w`
  steps/get_raw_transforms_test.sh  --cmd "$train_cmd" --num-jobs $nj \
    data/test_dev93 data/lang exp/tri2b_raw exp/tri2b/decode_tgpr_dev93 \
    exp/tri2b_raw/decode_tgpr_dev93_raw_trans


  # Train an LDA+MLLT system on top of the transformed raw MFCCs.
  steps/train_lda_mllt.sh --num-jobs 10 --cmd "$train_cmd" \
    --raw-transform-dir exp/tri2b_raw_trans_si84 \
     2500 15000 data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri3c || exit 1;

  # Test this system.  Note: this isn't our final number yet as we
  # haven't done 2nd pass of SAT.  Baseline for this is possibly tri2b...
  # but not sure if there's a truly comparable baseline.
  utils/mkgraph.sh data/lang_test_tgpr exp/tri3c exp/tri3c/graph_tgpr || exit 1;
  utils/decode.sh --num-jobs "$nj" --cmd "$decode_cmd" \
     --opts "--raw-transform-dir exp/tri2b_raw/decode_tgpr_dev93_raw_trans" \
     steps/decode_lda_mllt.sh exp/tri3c/graph_tgpr data/test_dev93 \
      exp/tri3c/decode_tgpr_dev93 || exit 1;

  # Align that LDA+MLLT system that used transformed MFCCs.
  steps/align_lda_mllt.sh  --num-jobs 10 --cmd "$train_cmd" \
    --raw-transform-dir exp/tri2b_raw_trans_si84 \
    --use-graphs data/train_si84 data/lang exp/tri3c exp/tri3c_ali_si84 || exit 1;

  # Train an LDA+MLLT+SAT system on top of transformed MFCCs.
  steps/train_lda_mllt_sat.sh  --num-jobs 10 --cmd "$train_cmd" \
    --raw-transform-dir exp/tri2b_raw_trans_si84 \
    2500 15000 data/train_si84 data/lang exp/tri3c_ali_si84 exp/tri4d || exit 1;

  nj=`ls exp/tri2b/decode_tgpr_dev93/lat.*.gz | wc -w`
  utils/mkgraph.sh data/lang_test_tgpr exp/tri4d exp/tri4d/graph_tgpr || exit 1;
  utils/decode.sh --cmd "$decode_cmd" \
    --num-jobs $nj --opts "--raw-transform-dir exp/tri2b_raw/decode_tgpr_dev93_raw_trans" \
   steps/decode_lda_mllt_sat.sh \
    exp/tri4d/graph_tgpr data/test_dev93 exp/tri4d/decode_tgpr_dev93 || exit 1;

)

# Train and test MMI (and boosted MMI) on tri2b system.
steps/make_denlats_lda_etc.sh --num-jobs 10 --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri2b_denlats_si84 || exit 1;
steps/train_lda_etc_mmi.sh --num-jobs 10  --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri2b_ali_si84 \
  exp/tri2b_denlats_si84 exp/tri2b exp/tri2b_mmi  || exit 1;

utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt.sh \
  exp/tri2b/graph_tgpr data/test_eval92 exp/tri2b_mmi/decode_tgpr_eval92  || exit 1;
steps/train_lda_etc_mmi.sh --num-jobs 10 --boost 0.1 --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri2b_denlats_si84 \
  exp/tri2b exp/tri2b_mmi_b0.1  || exit 1;
utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt.sh \
   exp/tri2b/graph_tgpr data/test_eval92 exp/tri2b_mmi_b0.1/decode_tgpr_eval92 || exit 1;

(
# HERE-- new
  steps/train_lda_etc_dmmi.sh --num-jobs 10  --cmd "$train_cmd" \
   data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri2b_denlats_si84 \
   exp/tri2b exp/tri2b_dmmi_-1.0_0.1

  utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt.sh \
    exp/tri2b/graph_tgpr data/test_eval92 exp/tri2b_fmmi_b0.1/decode_tgpr_eval92

  steps/train_lda_etc_dmmi.sh --num-jobs 10  --cmd "$train_cmd" \
   --num-boost -2.0 \
   data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri2b_denlats_si84 \
   exp/tri2b exp/tri2b_dmmi_-2.0_0.1

)
 # The next 3 commands train and test fMMI+MMI (on top of LDA+MLLT).
 steps/train_dubm_lda_etc.sh --silence-weight 0.5 \
   --num-jobs 10 --cmd "$train_cmd" 400 data/train_si84 \
   data/lang exp/tri2b_ali_si84 exp/dubm2b
 steps/train_lda_etc_mmi_fmmi.sh \
   --num-jobs 10 --boost 0.1 --cmd "$train_cmd" \
   data/train_si84 data/lang exp/tri2b_ali_si84 exp/dubm2b exp/tri2b_denlats_si84 \
   exp/tri2b exp/tri2b_fmmi_b0.1
 utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_fmpe.sh \
   exp/tri2b/graph_tgpr data/test_eval92 exp/tri2b_fmmi_b0.1/decode_tgpr_eval92


steps/train_lda_etc_mce.sh --cmd "$train_cmd" --num-jobs 10 data/train_si84 data/lang \
 exp/tri2b_ali_si84 exp/tri2b_denlats_si84 exp/tri2b exp/tri2b_mce || exit 1;
 utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt.sh \
   exp/tri2b/graph_tgpr data/test_eval92 exp/tri2b_mce/decode_tgpr_eval92 || exit 1;


# Train LDA+ET system.
steps/train_lda_et.sh --num-jobs 10 --cmd "$train_cmd" \
  2500 15000 data/train_si84 data/lang exp/tri1_ali_si84 exp/tri2c || exit 1;
utils/mkgraph.sh data/lang_test_tgpr exp/tri2c exp/tri2c/graph_tgpr || exit 1;
utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_et.sh exp/tri2c/graph_tgpr data/test_dev93 exp/tri2c/decode_tgpr_dev93 || exit 1;
utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_et_2pass.sh exp/tri2c/graph_tgpr data/test_dev93 exp/tri2c/decode_tgpr_dev93_2pass || exit 1;

# From 2b system, train 3b which is LDA + MLLT + SAT.
steps/train_lda_mllt_sat.sh  --num-jobs 10 --cmd "$train_cmd" \
  2500 15000 data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri3b || exit 1;
utils/mkgraph.sh data/lang_test_tgpr exp/tri3b exp/tri3b/graph_tgpr || exit 1;
utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh \
  exp/tri3b/graph_tgpr data/test_dev93 exp/tri3b/decode_tgpr_dev93 || exit 1;
utils/lmrescore.sh --cmd "$decode_cmd" data/lang_test_tgpr data/lang_test_tg \
  data/test_dev93 exp/tri3b/decode_tgpr_dev93 exp/tri3b/decode_tgpr_dev93_tg || exit 1;
utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh \
  exp/tri3b/graph_tgpr data/test_eval92 exp/tri3b/decode_tgpr_eval92 || exit 1;
utils/lmrescore.sh --cmd "$decode_cmd" data/lang_test_tgpr data/lang_test_tg \
  data/test_eval92 exp/tri3b/decode_tgpr_eval92 exp/tri3b/decode_tgpr_eval92_tg \
   || exit 1;

 # This step interpolates a small RNNLM (with weight 0.25) with the 4-gram LM.
 utils/rnnlmrescore.sh \
  --inv-acwt 17 \
  0.25 data/lang_test_bd_fg data/local/rnnlm/rnnlm.voc10000.hl30 data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm_0.25  \
  || exit 1;


# Trying the larger dictionary ("big-dict"/bd) + locally produced LM.
utils/mkgraph.sh data/lang_test_bd_tgpr exp/tri3b exp/tri3b/graph_bd_tgpr || exit 1;
utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh \
  exp/tri3b/graph_bd_tgpr data/test_eval92 exp/tri3b/decode_bd_tgpr_eval92 || exit 1;

utils/mkgraph.sh data/lang_test_bd_tgpr exp/tri3b exp/tri3b/graph_bd_tgpr || exit 1;
utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh \
  exp/tri3b/graph_bd_tgpr data/test_eval92 exp/tri3b/decode_bd_tgpr_eval92 || exit 1;
utils/lmrescore.sh --cmd "$decode_cmd" data/lang_test_bd_tgpr data/lang_test_bd_fg \
  data/test_eval92 exp/tri3b/decode_bd_tgpr_eval92 exp/tri3b/decode_bd_tgpr_eval92_fg \
   || exit 1;
utils/lmrescore.sh --cmd "$decode_cmd" data/lang_test_bd_tgpr data/lang_test_bd_tg \
  data/test_eval92 exp/tri3b/decode_bd_tgpr_eval92 exp/tri3b/decode_bd_tgpr_eval92_tg \
  || exit 1;

# The following two steps, which are a kind of side-branch, try mixing up
( # from the 3b system.  This is to demonstrate that script.
 steps/mixup_lda_etc.sh --num-jobs 10 --cmd "$train_cmd" \
   20000 data/train_si84 exp/tri3b exp/tri2b_ali_si84 exp/tri3b_20k || exit 1;
 utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh \
   exp/tri3b/graph_tgpr data/test_dev93 exp/tri3b_20k/decode_tgpr_dev93  || exit 1;
)


# From 3b system, align all si284 data.
steps/align_lda_mllt_sat.sh --num-jobs 10 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri3b exp/tri3b_ali_si284 || exit 1;

steps/train_lda_etc_quick.sh --num-jobs 10 --cmd "$train_cmd" \
   4200 40000 data/train_si284 data/lang exp/tri3b_ali_si284 exp/tri4b || exit 1;
utils/mkgraph.sh data/lang_test_tgpr exp/tri4b exp/tri4b/graph_tgpr || exit 1;
utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh \
  exp/tri4b/graph_tgpr data/test_dev93 exp/tri4b/decode_tgpr_dev93 || exit 1;
utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh \
  exp/tri4b/graph_tgpr data/test_eval92 exp/tri4b/decode_tgpr_eval92 || exit 1;

# Train and test MMI, and boosted MMI, on tri4b (LDA+MLLT+SAT on
# all the data).
# Making num-jobs 40 as want to keep them under 4 hours long (or will fail
# on regular queue at BUT).
steps/align_lda_mllt_sat.sh --num-jobs 40 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri4b exp/tri4b_ali_si284 || exit 1;
steps/make_denlats_lda_etc.sh --num-jobs 40 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri4b_ali_si284 exp/tri4b_denlats_si284 || exit 1;
steps/train_lda_etc_mmi.sh --num-jobs 40 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri4b_ali_si284 exp/tri4b_denlats_si284 \
  exp/tri4b exp/tri4b_mmi || exit 1;
utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_etc.sh exp/tri4b/graph_tgpr \
  data/test_dev93 exp/tri4b_mmi/decode_tgpr_dev93 exp/tri4b/decode_tgpr_dev93 \
   || exit 1;
steps/train_lda_etc_mmi.sh --boost 0.1 --num-jobs 40 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri4b_ali_si284 exp/tri4b_denlats_si284 \
  exp/tri4b exp/tri4b_mmi_b0.1 || exit 1;
utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_etc.sh exp/tri4b/graph_tgpr \
 data/test_dev93 exp/tri4b_mmi_b0.1/decode_tgpr_dev93 exp/tri4b/decode_tgpr_dev93 \
   || exit 1;
 utils/decode.sh --opts "--beam 15" --cmd "$decode_cmd" steps/decode_lda_etc.sh exp/tri4b/graph_tgpr data/test_dev93 exp/tri4b_mmi_b0.1/decode_tgpr_dev93_b15 exp/tri4b/decode_tgpr_dev93
utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_etc.sh exp/tri4b/graph_tgpr data/test_eval92 exp/tri4b_mmi_b0.1/decode_tgpr_eval92 exp/tri4b/decode_tgpr_eval92

 # Train fMMI+MMI system on top of 4b.
 steps/train_dubm_lda_etc.sh --silence-weight 0.5 \
   --num-jobs 40 --cmd "$train_cmd" 600 data/train_si284 \
   data/lang exp/tri4b_ali_si284 exp/dubm4b
 steps/train_lda_etc_mmi_fmmi.sh \
   --num-jobs 40 --boost 0.1 --cmd "$train_cmd" \
   data/train_si284 data/lang exp/tri4b_ali_si284 exp/dubm4b exp/tri4b_denlats_si284 \
   exp/tri4b exp/tri4b_fmmi_b0.1 
 utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_etc_fmpe.sh \
   exp/tri4b/graph_tgpr data/test_eval92 exp/tri4b_fmmi_b0.1/decode_tgpr_eval92 \
   exp/tri4b/decode_tgpr_eval92
 utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_etc_fmpe.sh \
   exp/tri4b/graph_tgpr data/test_dev93 exp/tri4b_fmmi_b0.1/decode_tgpr_dev93 \
   exp/tri4b/decode_tgpr_dev93



# Train UBM, for SGMM system on top of LDA+MLLT.
steps/train_ubm_lda_etc.sh --num-jobs 10 --cmd "$train_cmd" \
  400 data/train_si84 data/lang exp/tri2b_ali_si84 exp/ubm3c || exit 1;
steps/train_sgmm_lda_etc.sh --num-jobs 10 --cmd "$train_cmd" \
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
steps/align_lda_mllt_sat.sh --num-jobs 10 --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri3b exp/tri3b_ali_si84 || exit 1;
steps/train_ubm_lda_etc.sh --num-jobs 10 --cmd "$train_cmd" \
  400 data/train_si84 data/lang exp/tri3b_ali_si84 exp/ubm4b || exit 1;
steps/train_sgmm_lda_etc.sh  --num-jobs 10 --cmd "$train_cmd" \
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
  steps/mixup_sgmm_lda_etc.sh --num-jobs 10 --cmd "$train_cmd" \
     12500 data/train_si84 exp/sgmm4b exp/tri3b_ali_si84 exp/sgmm4b_12500 || exit 1;
  utils/decode.sh --cmd "$decode_cmd" steps/decode_sgmm_lda_etc.sh  \
    exp/sgmm4b/graph_tgpr data/test_eval92 exp/sgmm4b_12500/decode_tgpr_eval92 exp/tri3b/decode_tgpr_eval92 || exit 1;
 # note: taking it up to 150k made it worse again [8.63->8.56->8.72 ... this was before some
 # decoding-script changes so these results not the same as in RESULTS file.]
  # increasing phone dim but not #substates..
  steps/mixup_sgmm_lda_etc.sh --num-jobs 10 --cmd "$train_cmd" --increase-phone-dim 50 \
    10000 data/train_si84 exp/sgmm4b exp/tri3b_ali_si84 exp/sgmm4b_50 || exit 1;
  utils/decode.sh --cmd "$decode_cmd" steps/decode_sgmm_lda_etc.sh  \
    exp/sgmm4b/graph_tgpr data/test_eval92 exp/sgmm4b_50/decode_tgpr_eval92 \
   exp/tri3b/decode_tgpr_eval92 || exit 1;

# Align 3b system with si284 data and num-jobs = 20; we'll train an LDA+MLLT+SAT system on si284 from this.
# This is 4c.  c.f. 4b which is "quick" training.

steps/align_lda_mllt_sat.sh --num-jobs 20 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri3b exp/tri3b_ali_si284_20 || exit 1;
steps/train_lda_mllt_sat.sh --num-jobs 20 --cmd "$train_cmd" \
  4200 40000 data/train_si284 data/lang exp/tri3b_ali_si284_20 exp/tri4c || exit 1;
utils/mkgraph.sh data/lang_test_tgpr exp/tri4c exp/tri4c/graph_tgpr || exit 1;
utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh \
  exp/tri4c/graph_tgpr data/test_dev93 exp/tri4c/decode_tgpr_dev93 || exit 1;


( # Try mixing up the tri4c system further.
 steps/mixup_lda_etc.sh --num-jobs 20 --cmd "$train_cmd" \
  50000 data/train_si284 exp/tri4c exp/tri3b_ali_si284_20 exp/tri4c_50k || exit 1;
 utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh \
  exp/tri4c/graph_tgpr data/test_dev93 exp/tri4c_50k/decode_tgpr_dev93 || exit 1;

 steps/mixup_lda_etc.sh --num-jobs 20 --cmd "$train_cmd" \
  75000 data/train_si284 exp/tri4c_50k exp/tri3b_ali_si284_20 exp/tri4c_75k || exit 1;
 utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh \
   exp/tri4c/graph_tgpr data/test_dev93 exp/tri4c_75k/decode_tgpr_dev93 || exit 1;

 steps/mixup_lda_etc.sh --num-jobs 20 --cmd "$train_cmd" \
  100000 data/train_si284 exp/tri4c_75k exp/tri3b_ali_si284_20 exp/tri4c_100k || exit 1;
 utils/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt_sat.sh \
  exp/tri4c/graph_tgpr data/test_dev93 exp/tri4c_100k/decode_tgpr_dev93 || exit 1;
)


# Train SGMM on top of LDA+MLLT+SAT, on all SI-284 data.  C.f. 4b which was
# just on SI-84.
steps/train_ubm_lda_etc.sh --num-jobs 20 --cmd "$train_cmd" \
  600 data/train_si284 data/lang exp/tri3b_ali_si284_20 exp/ubm4c || exit 1;
steps/train_sgmm_lda_etc.sh  --num-jobs 20 --cmd "$train_cmd" \
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
steps/train_sgmm_lda_etc.sh  --num-jobs 20 --cmd "$train_cmd" \
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
