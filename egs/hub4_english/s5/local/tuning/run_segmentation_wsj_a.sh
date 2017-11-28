#!/bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0

set -e
set -o pipefail

# This script demonstrates how to use out-of-domain WSJ models to segment long
# audio recordings of HUB4 with raw unaligned transcripts into short segments
# with aligned transcripts for training new ASR models. 

# The overall procedure is as follow:
# 1) Train a GMM on out-of-domain WSJ corpus
# 2) Decode broadcast news recordings (HUB4) with WSJ GMM and 4-gram biased LM 
#    trained on the raw unprocessed transcript. 
# 3) Use the CTM output to segment the recordings keep the best matched
#    audio and text.
# 4) Train an in-domain GMM on the above data. 
# 5) Repeat steps 2, 3 and 4 using the new in-domain GMM.
# 6) Re-segment the data retaining only the "clean" part of the data.

# See the script steps/cleanup/segment_long_utterances.sh for details about 
# audio-transcript alignment (Step 2, 3)
# See the script steps/cleanup/clean_and_segment_data.sh for details about 
# cleaning up transcripts (Step 6)

# WSJ models (From step 1)
# %WER 29.5 | 728 32834 | 73.1 17.7 9.2 2.6 29.5 92.2 | exp/wsj_tri3/decode_nosp_test_eval97.pem_rescore/score_16_0.0/eval97.pem.ctm.filt.sys
# %WER 30.4 | 728 32834 | 72.3 18.3 9.4 2.7 30.4 92.3 | exp/wsj_tri3/decode_nosp_test_eval97.pem/score_16_0.0/eval97.pem.ctm.filt.sys

# In-domain GMM (From step 4)
# %WER 19.8 | 728 32834 | 82.1 12.6 5.3 1.9 19.8 85.9 | exp/tri4_a/decode_nosp_eval97.pem_rescore/score_15_1.0/eval97.pem.ctm.filt.sys
# %WER 20.9 | 728 32834 | 81.2 13.5 5.3 2.1 20.9 86.5 | exp/tri4_a/decode_nosp_eval97.pem/score_14_0.0/eval97.pem.ctm.filt.sys

# In-domain GMM (From step 5)
# %WER 20.4 | 728 32834 | 81.7 13.1 5.2 2.1 20.4 86.1 | exp/tri4_2a/decode_nosp_eval97.pem_rescore/score_14_0.0/eval97.pem.ctm.filt.sys
# %WER 21.3 | 728 32834 | 80.7 13.7 5.6 2.0 21.3 87.1 | exp/tri4_2a/decode_nosp_eval97.pem/score_15_1.0/eval97.pem.ctm.filt.sys

# GMM trained on cleaned transcripts (From step 6)
# %WER 19.1 | 728 32834 | 83.1 12.2 4.7 2.2 19.1 85.0 | exp/tri5_2a_cleaned/decode_nosp_eval97.pem_rescore/score_13_0.0/eval97.pem.ctm.filt.sys
# %WER 20.2 | 728 32834 | 81.9 13.0 5.1 2.1 20.2 87.1 | exp/tri5_2a_cleaned/decode_nosp_eval97.pem/score_14_0.0/eval97.pem.ctm.filt.sys

# Oracle HUB4 transcripts
# %WER 18.0 | 728 32834 | 83.9 11.7 4.3 2.0 18.0 85.9 | exp/tri4/decode_nosp_eval97.pem_rescore/score_14_0.0/eval97.pem.ctm.filt.sys
# %WER 19.3 | 728 32834 | 82.9 12.6 4.6 2.2 19.3 86.8 | exp/tri4/decode_nosp_eval97.pem/score_13_0.0/eval97.pem.ctm.filt.sys

. ./cmd.sh
. ./path.sh

segment_stage=-8
nj=40
reco_nj=80
affix=a
new_affix=2a

. utils/parse_options.sh

###############################################################################
## Simulate unsegmented HUB4 data directory.
###############################################################################
utils/data/convert_data_dir_to_whole.sh data/train data/train_long

steps/make_mfcc.sh --cmd "$train_cmd --max-jobs-run 40" --nj $reco_nj \
  data/train_long exp/make_mfcc/train_long mfcc || exit 1
steps/compute_cmvn_stats.sh data/train_long \
  exp/make_mfcc/train_long mfcc
utils/fix_data_dir.sh data/train_long

###############################################################################
## Train GMM on out-of-domain WSJ corpus 
###############################################################################

steps/train_mono.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
  data/train_si84_2kshort data/lang_nosp exp/wsj_mono0a || exit 1;

steps/align_si.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
  data/train_si84 data/lang_nosp exp/wsj_mono0a exp/wsj_mono0a_ali_si84 || exit 1;

steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2500 15000 \
  data/train_si84 data/lang_nosp exp/wsj_mono0a_ali_si84 exp/wsj_tri1 || exit 1;

steps/align_si.sh --nj $nj --cmd "$train_cmd" \
  data/train_si284 data/lang_nosp exp/wsj_tri1 exp/wsj_tri1_ali_si284 || exit 1;

steps/train_lda_mllt.sh --cmd "$train_cmd" \
  --splice-opts "--left-context=3 --right-context=3" 4000 42000 \
  data/train_si284 data/lang_nosp exp/wsj_tri1_ali_si284 exp/wsj_tri2 || exit 1;

steps/align_si.sh --nj $nj --cmd "$train_cmd" \
  data/train_si284 data/lang_nosp exp/wsj_tri2 exp/wsj_tri2_ali_si284 || exit 1

steps/train_sat.sh --cmd "$train_cmd" \
  4000 42000 \
  data/train_si284 data/lang_nosp exp/wsj_tri2_ali_si284 exp/wsj_tri3

###############################################################################
# Segment long HUB4 recordings and retrieve transcript using 
# Smith-Waterman alignment.
# Use a SAT model trained on train_si284 (wsj_tri3) as seed model for decoding.
###############################################################################

steps/cleanup/segment_long_utterances.sh --cmd "$train_cmd" \
  --stage $segment_stage --nj $reco_nj \
  --max-bad-proportion 0.5 --align-full-hyp false \
  exp/wsj_tri3 data/lang_nosp data/train_long data/train_long/text \
  data/train_reseg_${affix} exp/segment_long_utts_${affix}_train

steps/compute_cmvn_stats.sh data/train_reseg_${affix} \
  exp/make_mfcc/train_reseg_${affix} mfcc
utils/fix_data_dir.sh data/train_reseg_${affix}

utils/data/modify_speaker_info.sh data/train_reseg_${affix} \
  data/train_reseg_${affix}_spk30sec
steps/compute_cmvn_stats.sh data/train_reseg_${affix}_spk30sec \
  exp/make_mfcc/train_reseg_${affix}_spk30sec mfcc
utils/fix_data_dir.sh data/train_reseg_${affix}_spk30sec

###############################################################################
# Train new in-domain GMM (tri4_a) on retrieved transcripts.
###############################################################################

steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
  data/train_reseg_${affix}_spk30sec data/lang_nosp \
  exp/wsj_tri3 exp/wsj_tri3_ali_train_reseg_${affix}

steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
  data/train_reseg_${affix}_spk30sec data/lang_nosp \
  exp/wsj_tri3_ali_train_reseg_${affix} exp/tri3_${affix} 

steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
  data/train_reseg_${affix}_spk30sec data/lang_nosp exp/tri3_${affix} exp/tri3_${affix}_ali

steps/train_sat.sh --cmd "$train_cmd" 5000 100000 \
  data/train_reseg_${affix}_spk30sec data/lang_nosp exp/tri3_${affix}_ali exp/tri4_${affix}

utils/mkgraph.sh data/lang_nosp_test exp/tri4_${affix}/{,graph_nosp_test}
for dset in eval97.pem; do
  this_nj=`cat data/$dset/spk2utt | wc -l`
  if [ $this_nj -gt 20 ]; then
    this_nj=20
  fi
  steps/decode_fmllr.sh --nj $this_nj --cmd "$decode_cmd" --num-threads 4 \
    exp/tri4_${affix}/graph_nosp_test data/$dset exp/tri4_${affix}/decode_nosp_${dset}
  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
    data/lang_nosp_test data/lang_nosp_test_rescore \
    data/${dset} exp/tri4_${affix}/decode_nosp_${dset} \
    exp/tri4_${affix}/decode_nosp_${dset}_rescore
done

###############################################################################
# Segment long HUB4 recordings and retrieve transcript using 
# Smith-Waterman alignment.
# Use in-domain SAT model (tri4_a) as seed model for decoding.
###############################################################################

steps/cleanup/segment_long_utterances.sh --cmd "$train_cmd" \
  --stage $segment_stage --nj $reco_nj \
  --max-bad-proportion 0.5 --align-full-hyp false \
  exp/tri4_${affix} data/lang_nosp data/train_long data/train_long/text \
  data/train_reseg_${new_affix} exp/segment_long_utts_${new_affix}_train

steps/compute_cmvn_stats.sh data/train_reseg_${new_affix} \
  exp/make_mfcc/train_reseg_${new_affix} mfcc
utils/fix_data_dir.sh data/train_reseg_${new_affix}

###############################################################################
# Train new in-domain GMM (tri4_2a) on retrieved transcripts.
###############################################################################

steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
  data/train_reseg_${new_affix} data/lang_nosp \
  exp/tri4_${affix} exp/tri4_${affix}_ali_train_reseg_${new_affix}

steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
  data/train_reseg_${new_affix} data/lang_nosp \
  exp/tri4_${affix}_ali_train_reseg_${new_affix} exp/tri4_${new_affix} 

utils/mkgraph.sh data/lang_nosp_test exp/tri4_${new_affix}/{,graph_nosp_test}
for dset in eval97.pem; do
  this_nj=`cat data/$dset/spk2utt | wc -l`
  if [ $this_nj -gt 20 ]; then
    this_nj=20
  fi
  steps/decode_fmllr.sh --nj $this_nj --cmd "$decode_cmd" --num-threads 4 \
    exp/tri4_${new_affix}/graph_nosp_test data/$dset exp/tri4_${new_affix}/decode_nosp_${dset}
  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
    data/lang_nosp_test data/lang_nosp_test_rescore \
    data/${dset} exp/tri4_${new_affix}/decode_nosp_${dset} \
    exp/tri4_${new_affix}/decode_nosp_${dset}_rescore
done

###############################################################################
# Cleanup transcripts
# Use in-domain SAT model (tri4_2a) as seed model for decoding.
###############################################################################

cleanup_stage=-1
cleanup_affix=cleaned
srcdir=exp/tri4_${new_affix}
cleaned_data=data/train_reseg_${new_affix}_${cleanup_affix}
dir=${srcdir}_${cleanup_affix}_work
cleaned_dir=${srcdir}_${cleanup_affix}

steps/cleanup/clean_and_segment_data.sh --stage $cleanup_stage --nj 80 \
  --cmd "$train_cmd" \
  data/train_reseg_${new_affix} data/lang_nosp \
  $srcdir $dir $cleaned_data

###############################################################################
# Train new in-domain GMM (tri4_2a) on cleaned-up transcripts.
###############################################################################

steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
  $cleaned_data data/lang_nosp $srcdir ${srcdir}_ali_${cleanup_affix}

steps/train_sat.sh --cmd "$train_cmd" \
  5000 100000 $cleaned_data data/lang_nosp \
  ${srcdir}_ali_${cleanup_affix} exp/tri5_${new_affix}_${cleanup_affix}

utils/mkgraph.sh data/lang_nosp_test \
  exp/tri5_${new_affix}_${cleanup_affix}/{,graph_nosp_test}
for dset in eval97.pem; do
  this_nj=`cat data/$dset/spk2utt | wc -l`
  if [ $this_nj -gt 20 ]; then
    this_nj=20
  fi
  steps/decode_fmllr.sh --nj $this_nj --cmd "$decode_cmd" --num-threads 4 \
    exp/tri5_${new_affix}_${cleanup_affix}/graph_nosp_test data/$dset \
    exp/tri5_${new_affix}_${cleanup_affix}/decode_nosp_${dset}
  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
    data/lang_nosp_test data/lang_nosp_test_rescore \
    data/${dset} exp/tri5_${new_affix}_${cleanup_affix}/decode_nosp_${dset} \
    exp/tri5_${new_affix}_${cleanup_affix}/decode_nosp_${dset}_rescore
done

exit 0
