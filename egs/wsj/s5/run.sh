#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

#wsj0=/ais/gobi2/speech/WSJ/csr_?_senn_d?
#wsj1=/ais/gobi2/speech/WSJ/csr_senn_d?

#wsj0=/mnt/matylda2/data/WSJ0
#wsj1=/mnt/matylda2/data/WSJ1

#wsj0=/data/corpora0/LDC93S6B
#wsj1=/data/corpora0/LDC94S13B

wsj0=/export/corpora5/LDC/LDC93S6B
wsj1=/export/corpora5/LDC/LDC94S13B

local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?  || exit 1;

# Sometimes, we have seen WSJ distributions that do not have subdirectories
# like '11-13.1', but instead have 'doc', 'si_et_05', etc. directly under the
# wsj0 or wsj1 directories. In such cases, try the following:
#
# corpus=/exports/work/inf_hcrc_cstr_general/corpora/wsj
# local/cstr_wsj_data_prep.sh $corpus
# rm data/local/dict/lexiconp.txt
# $corpus must contain a 'wsj0' and a 'wsj1' subdirectory for this to work.
#
# "nosp" refers to the dictionary before silence probabilities and pronunciation
# probabilities are added.
local/wsj_prepare_dict.sh --dict-suffix "_nosp" || exit 1;

utils/prepare_lang.sh data/local/dict_nosp \
  "<SPOKEN_NOISE>" data/local/lang_tmp_nosp data/lang_nosp || exit 1;

local/wsj_format_data.sh --lang-suffix "_nosp" || exit 1;

 # We suggest to run the next three commands in the background,
 # as they are not a precondition for the system building and
 # most of the tests: these commands build a dictionary
 # containing many of the OOVs in the WSJ LM training data,
 # and an LM trained directly on that data (i.e. not just
 # copying the arpa files from the disks from LDC).
 # Caution: the commands below will only work if $decode_cmd
 # is setup to use qsub.  Else, just remove the --cmd option.
 # NOTE: If you have a setup corresponding to the older cstr_wsj_data_prep.sh style,
 # use local/cstr_wsj_extend_dict.sh --dict-suffix "_nosp" $corpus/wsj1/doc/ instead.
  (
   local/wsj_extend_dict.sh --dict-suffix "_nosp" $wsj1/13-32.1  && \
   utils/prepare_lang.sh data/local/dict_nosp_larger \
     "<SPOKEN_NOISE>" data/local/lang_tmp_nosp_larger data/lang_nosp_bd && \
   local/wsj_train_lms.sh --dict-suffix "_nosp" &&
   local/wsj_format_local_lms.sh --lang-suffix "_nosp" # &&
  ) &

# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.

for x in test_eval92 test_eval93 test_dev93 train_si284; do
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 20 data/$x || exit 1;
  steps/compute_cmvn_stats.sh data/$x || exit 1;
done

utils/subset_data_dir.sh --first data/train_si284 7138 data/train_si84 || exit 1

# Now make subset with the shortest 2k utterances from si-84.
utils/subset_data_dir.sh --shortest data/train_si84 2000 data/train_si84_2kshort || exit 1;

# Now make subset with half of the data from si-84.
utils/subset_data_dir.sh data/train_si84 3500 data/train_si84_half || exit 1;


# Note: the --boost-silence option should probably be omitted by default
# for normal setups.  It doesn't always help. [it's to discourage non-silence
# models from modeling silence.]
steps/train_mono.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
  data/train_si84_2kshort data/lang_nosp exp/mono0a || exit 1;

(
 utils/mkgraph.sh data/lang_nosp_test_tgpr \
   exp/mono0a exp/mono0a/graph_nosp_tgpr && \
 steps/decode.sh --nj 10 --cmd "$decode_cmd" exp/mono0a/graph_nosp_tgpr \
   data/test_dev93 exp/mono0a/decode_nosp_tgpr_dev93 && \
 steps/decode.sh --nj 8 --cmd "$decode_cmd" exp/mono0a/graph_nosp_tgpr \
   data/test_eval92 exp/mono0a/decode_nosp_tgpr_eval92
) &

steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
  data/train_si84_half data/lang_nosp exp/mono0a exp/mono0a_ali || exit 1;

steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 \
  data/train_si84_half data/lang_nosp exp/mono0a_ali exp/tri1 || exit 1;

while [ ! -f data/lang_nosp_test_tgpr/tmp/LG.fst ] || \
   [ -z data/lang_nosp_test_tgpr/tmp/LG.fst ]; do
  sleep 20;
done
sleep 30;
# or the mono mkgraph.sh might be writing
# data/lang_test_tgpr/tmp/LG.fst which will cause this to fail.

utils/mkgraph.sh data/lang_nosp_test_tgpr \
  exp/tri1 exp/tri1/graph_nosp_tgpr || exit 1;

steps/decode.sh --nj 10 --cmd "$decode_cmd" exp/tri1/graph_nosp_tgpr \
  data/test_dev93 exp/tri1/decode_nosp_tgpr_dev93 || exit 1;
steps/decode.sh --nj 8 --cmd "$decode_cmd" exp/tri1/graph_nosp_tgpr \
  data/test_eval92 exp/tri1/decode_nosp_tgpr_eval92 || exit 1;

# test various modes of LM rescoring (4 is the default one).
# This is just confirming they're equivalent.
for mode in 1 2 3 4; do
  steps/lmrescore.sh --mode $mode --cmd "$decode_cmd" \
    data/lang_nosp_test_{tgpr,tg} data/test_dev93 \
    exp/tri1/decode_nosp_tgpr_dev93 \
    exp/tri1/decode_nosp_tgpr_dev93_tg$mode  || exit 1;
done


## the following command demonstrates how to get lattices that are
## "word-aligned" (arcs coincide with words, with boundaries in the right
## place).
#sil_label=`grep '!SIL' data/lang_nosp_test_tgpr/words.txt | awk '{print $2}'`
#steps/word_align_lattices.sh --cmd "$train_cmd" --silence-label $sil_label \
#  data/lang_nosp_test_tgpr exp/tri1/decode_nosp_tgpr_dev93 \
#  exp/tri1/decode_nosp_tgpr_dev93_aligned || exit 1;

steps/align_si.sh --nj 10 --cmd "$train_cmd" \
  data/train_si84 data/lang_nosp exp/tri1 exp/tri1_ali_si84 || exit 1;

steps/train_lda_mllt.sh --cmd "$train_cmd" \
  --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
  data/train_si84 data/lang_nosp exp/tri1_ali_si84 exp/tri2b || exit 1;

utils/mkgraph.sh data/lang_nosp_test_tgpr \
  exp/tri2b exp/tri2b/graph_nosp_tgpr || exit 1;
steps/decode.sh --nj 10 --cmd "$decode_cmd" exp/tri2b/graph_nosp_tgpr \
  data/test_dev93 exp/tri2b/decode_nosp_tgpr_dev93 || exit 1;
steps/decode.sh --nj 8 --cmd "$decode_cmd" exp/tri2b/graph_nosp_tgpr \
  data/test_eval92 exp/tri2b/decode_nosp_tgpr_eval92 || exit 1;

# At this point, you could run the example scripts that show how VTLN works.
# We haven't included this in the default recipes yet.
# local/run_vtln.sh --lang-suffix "_nosp"
# local/run_vtln2.sh --lang-suffix "_nosp"

# Now, with dev93, compare lattice rescoring with biglm decoding,
# going from tgpr to tg.  Note: results are not the same, even though they should
# be, and I believe this is due to the beams not being wide enough.  The pruning
# seems to be a bit too narrow in the current scripts (got at least 0.7% absolute
# improvement from loosening beams from their current values).

steps/decode_biglm.sh --nj 10 --cmd "$decode_cmd" \
  exp/tri2b/graph_nosp_tgpr data/lang_test_{tgpr,tg}/G.fst \
  data/test_dev93 exp/tri2b/decode_nosp_tgpr_dev93_tg_biglm

# baseline via LM rescoring of lattices.
steps/lmrescore.sh --cmd "$decode_cmd" \
  data/lang_nosp_test_tgpr/ data/lang_nosp_test_tg/ \
  data/test_dev93 exp/tri2b/decode_nosp_tgpr_dev93 \
  exp/tri2b/decode_nosp_tgpr_dev93_tg || exit 1;

# Trying Minimum Bayes Risk decoding (like Confusion Network decoding):
mkdir exp/tri2b/decode_nosp_tgpr_dev93_tg_mbr
cp exp/tri2b/decode_nosp_tgpr_dev93_tg/lat.*.gz \
  exp/tri2b/decode_nosp_tgpr_dev93_tg_mbr
local/score_mbr.sh --cmd "$decode_cmd" \
 data/test_dev93/ data/lang_nosp_test_tgpr/ \
 exp/tri2b/decode_nosp_tgpr_dev93_tg_mbr

# This script trains a delta+delta-delta system.  It's not really recommended or
# necessary, but it does contain a demonstration of the decode_fromlats.sh
# script which isn't used elsewhere.
# local/run_deltas.sh

# Align tri2b system with si84 data.
steps/align_si.sh  --nj 10 --cmd "$train_cmd" \
  --use-graphs true data/train_si84 \
  data/lang_nosp exp/tri2b exp/tri2b_ali_si84  || exit 1;

local/run_mmi_tri2b.sh --lang-suffix "_nosp"

# From 2b system, train 3b which is LDA + MLLT + SAT.
steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
  data/train_si84 data/lang_nosp exp/tri2b_ali_si84 exp/tri3b || exit 1;
utils/mkgraph.sh data/lang_nosp_test_tgpr \
  exp/tri3b exp/tri3b/graph_nosp_tgpr || exit 1;
steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
  exp/tri3b/graph_nosp_tgpr data/test_dev93 \
  exp/tri3b/decode_nosp_tgpr_dev93 || exit 1;
steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
  exp/tri3b/graph_nosp_tgpr data/test_eval92 \
  exp/tri3b/decode_nosp_tgpr_eval92 || exit 1;

# At this point you could run the command below; this gets
# results that demonstrate the basis-fMLLR adaptation (adaptation
# on small amounts of adaptation data).
local/run_basis_fmllr.sh --lang-suffix "_nosp"

steps/lmrescore.sh --cmd "$decode_cmd" \
  data/lang_nosp_test_tgpr data/lang_nosp_test_tg \
  data/test_dev93 exp/tri3b/decode_nosp_tgpr_dev93 \
  exp/tri3b/decode_nosp_tgpr_dev93_tg || exit 1;
steps/lmrescore.sh --cmd "$decode_cmd" \
  data/lang_nosp_test_tgpr data/lang_nosp_test_tg \
  data/test_eval92 exp/tri3b/decode_nosp_tgpr_eval92 \
  exp/tri3b/decode_nosp_tgpr_eval92_tg || exit 1;

# Trying the larger dictionary ("big-dict"/bd) + locally produced LM.
utils/mkgraph.sh data/lang_nosp_test_bd_tgpr \
  exp/tri3b exp/tri3b/graph_nosp_bd_tgpr || exit 1;

steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 8 \
  exp/tri3b/graph_nosp_bd_tgpr data/test_eval92 \
  exp/tri3b/decode_nosp_bd_tgpr_eval92 || exit 1;
steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 10 \
  exp/tri3b/graph_nosp_bd_tgpr data/test_dev93 \
  exp/tri3b/decode_nosp_bd_tgpr_dev93 || exit 1;

# Example of rescoring with ConstArpaLm.
steps/lmrescore_const_arpa.sh \
  --cmd "$decode_cmd" data/lang_nosp_test_bd_{tgpr,fgconst} \
  data/test_eval92 exp/tri3b/decode_nosp_bd_tgpr_eval92{,_fgconst} || exit 1;

steps/lmrescore.sh --cmd "$decode_cmd" \
  data/lang_nosp_test_bd_tgpr data/lang_nosp_test_bd_fg \
  data/test_eval92 exp/tri3b/decode_nosp_bd_tgpr_eval92 \
  exp/tri3b/decode_nosp_bd_tgpr_eval92_fg || exit 1;
steps/lmrescore.sh --cmd "$decode_cmd" \
  data/lang_nosp_test_bd_tgpr data/lang_nosp_test_bd_tg \
  data/test_eval92 exp/tri3b/decode_nosp_bd_tgpr_eval92 \
  exp/tri3b/decode_nosp_bd_tgpr_eval92_tg || exit 1;

# The following two steps, which are a kind of side-branch, try mixing up
( # from the 3b system.  This is to demonstrate that script.
 steps/mixup.sh --cmd "$train_cmd" \
   20000 data/train_si84 data/lang_nosp exp/tri3b exp/tri3b_20k || exit 1;
 steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 10 \
   exp/tri3b/graph_nosp_tgpr data/test_dev93 \
   exp/tri3b_20k/decode_nosp_tgpr_dev93  || exit 1;
)

# From 3b system, align all si284 data.
steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
  data/train_si284 data/lang_nosp exp/tri3b exp/tri3b_ali_si284 || exit 1;


# From 3b system, train another SAT system (tri4a) with all the si284 data.

steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
  data/train_si284 data/lang_nosp exp/tri3b_ali_si284 exp/tri4a || exit 1;
(
 utils/mkgraph.sh data/lang_nosp_test_tgpr \
   exp/tri4a exp/tri4a/graph_nosp_tgpr || exit 1;
 steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
   exp/tri4a/graph_nosp_tgpr data/test_dev93 \
   exp/tri4a/decode_nosp_tgpr_dev93 || exit 1;
 steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
   exp/tri4a/graph_nosp_tgpr data/test_eval92 \
   exp/tri4a/decode_nosp_tgpr_eval92 || exit 1;
) &


# This step is just to demonstrate the train_quick.sh script, in which we
# initialize the GMMs from the old system's GMMs.
steps/train_quick.sh --cmd "$train_cmd" 4200 40000 \
  data/train_si284 data/lang_nosp exp/tri3b_ali_si284 exp/tri4b || exit 1;

(
 utils/mkgraph.sh data/lang_nosp_test_tgpr \
   exp/tri4b exp/tri4b/graph_nosp_tgpr || exit 1;
 steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
   exp/tri4b/graph_nosp_tgpr data/test_dev93 \
   exp/tri4b/decode_nosp_tgpr_dev93 || exit 1;
 steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
  exp/tri4b/graph_nosp_tgpr data/test_eval92 \
  exp/tri4b/decode_nosp_tgpr_eval92 || exit 1;

 utils/mkgraph.sh data/lang_nosp_test_bd_tgpr \
   exp/tri4b exp/tri4b/graph_nosp_bd_tgpr || exit 1;
 steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
   exp/tri4b/graph_nosp_bd_tgpr data/test_dev93 \
   exp/tri4b/decode_nosp_bd_tgpr_dev93 || exit 1;
 steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
  exp/tri4b/graph_nosp_bd_tgpr data/test_eval92 \
  exp/tri4b/decode_nosp_bd_tgpr_eval92 || exit 1;
) &

# Silprob for normal lexicon.
steps/get_prons.sh --cmd "$train_cmd" \
  data/train_si284 data/lang_nosp exp/tri4b || exit 1;
utils/dict_dir_add_pronprobs.sh --max-normalize true \
  data/local/dict_nosp \
  exp/tri4b/pron_counts_nowb.txt exp/tri4b/sil_counts_nowb.txt \
  exp/tri4b/pron_bigram_counts_nowb.txt data/local/dict || exit 1

utils/prepare_lang.sh data/local/dict \
  "<SPOKEN_NOISE>" data/local/lang_tmp data/lang || exit 1;

for lm_suffix in bg bg_5k tg tg_5k tgpr tgpr_5k; do
  mkdir -p data/lang_test_${lm_suffix}
  cp -r data/lang/* data/lang_test_${lm_suffix}/ || exit 1;
  rm -rf data/lang_test_${lm_suffix}/tmp
  cp data/lang_nosp_test_${lm_suffix}/G.* data/lang_test_${lm_suffix}/
done

# Silprob for larger lexicon.
utils/dict_dir_add_pronprobs.sh --max-normalize true \
  data/local/dict_nosp_larger \
  exp/tri4b/pron_counts_nowb.txt exp/tri4b/sil_counts_nowb.txt \
  exp/tri4b/pron_bigram_counts_nowb.txt data/local/dict_larger || exit 1

utils/prepare_lang.sh data/local/dict_larger \
  "<SPOKEN_NOISE>" data/local/lang_tmp_larger data/lang_bd || exit 1;

for lm_suffix in tgpr tgconst tg fgpr fgconst fg; do
  mkdir -p data/lang_test_bd_${lm_suffix}
  cp -r data/lang_bd/* data/lang_test_bd_${lm_suffix}/ || exit 1;
  rm -rf data/lang_test_bd_${lm_suffix}/tmp
  cp data/lang_nosp_test_bd_${lm_suffix}/G.* data/lang_test_bd_${lm_suffix}/
done

(
 utils/mkgraph.sh data/lang_test_tgpr exp/tri4b exp/tri4b/graph_tgpr || exit 1;
 steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
   exp/tri4b/graph_tgpr data/test_dev93 exp/tri4b/decode_tgpr_dev93 || exit 1;
 steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
  exp/tri4b/graph_tgpr data/test_eval92 exp/tri4b/decode_tgpr_eval92 || exit 1;

 utils/mkgraph.sh data/lang_test_bd_tgpr \
   exp/tri4b exp/tri4b/graph_bd_tgpr || exit 1;
 steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
   exp/tri4b/graph_bd_tgpr data/test_dev93 \
   exp/tri4b/decode_bd_tgpr_dev93 || exit 1;
 steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
  exp/tri4b/graph_bd_tgpr data/test_eval92 \
  exp/tri4b/decode_bd_tgpr_eval92 || exit 1;
) &


# Train and test MMI, and boosted MMI, on tri4b (LDA+MLLT+SAT on
# all the data).  Use 30 jobs.
steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri4b exp/tri4b_ali_si284 || exit 1;

# These demonstrate how to build a sytem usable for online-decoding with the nnet2 setup.
# (see local/run_nnet2.sh for other, non-online nnet2 setups).
local/online/run_nnet2.sh
local/online/run_nnet2_baseline.sh
local/online/run_nnet2_discriminative.sh

# Demonstration of RNNLM rescoring on TDNN models. We comment this out by
# default.
# local/run_rnnlms.sh

local/run_mmi_tri4b.sh

#local/run_nnet2.sh

# You probably want to run the sgmm2 recipe as it's generally a bit better:
local/run_sgmm2.sh

# We demonstrate MAP adaptation of GMMs to gender-dependent systems here.  This also serves
# as a generic way to demonstrate MAP adaptation to different domains.
# local/run_gender_dep.sh

# You probably want to run the hybrid recipe as it is complementary:
local/nnet/run_dnn.sh

# The following demonstrate how to re-segment long audios.
# local/run_segmentation.sh

# The next two commands show how to train a bottleneck network based on the nnet2 setup,
# and build an SGMM system on top of it.
#local/run_bnf.sh
#local/run_bnf_sgmm.sh


# You probably want to try KL-HMM
#local/run_kl_hmm.sh

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

# # A couple of nnet3 recipes:
# local/nnet3/run_tdnn_baseline.sh  # designed for exact comparison with nnet2 recipe
# local/nnet3/run_tdnn.sh  # better absolute results
# local/nnet3/run_lstm.sh  # lstm recipe
# bidirectional lstm recipe
# local/nnet3/run_lstm.sh --affix bidirectional \
#	                  --lstm-delay " [-1,1] [-2,2] [-3,3] " \
#                         --label-delay 0 \
#                         --cell-dim 640 \
#                         --recurrent-projection-dim 128 \
#                         --non-recurrent-projection-dim 128 \
#                         --chunk-left-context 40 \
#                         --chunk-right-context 40
