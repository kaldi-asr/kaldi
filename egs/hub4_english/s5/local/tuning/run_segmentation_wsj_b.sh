#!/bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0

set -e
set -o pipefail

# This script demonstrates how to re-segment long audios into short segments.
# The basic idea is to decode with an existing out-of-domain WSJ GMM model, 
# and a 4-gram language model built from the reference, and then work out the
# segmentation from a ctm like file. This is used to build a stage 1 model
# that is used to decode and re-segment the long audio again to train a 
# stage 2 model. This is followed by a clean-up stage to get cleaned 
# transcripts.
# This is similar to _a but aligns full hypothesis with reference.



# Results using WSJ models
%WER 29.5 | 728 32834 | 73.1 17.7 9.2 2.6 29.5 92.2 | exp/wsj_tri3/decode_nosp_test_eval97.pem_rescore/score_16_0.0/eval97.pem.ctm.filt.sys
%WER 30.4 | 728 32834 | 72.3 18.3 9.4 2.7 30.4 92.3 | exp/wsj_tri3/decode_nosp_test_eval97.pem/score_16_0.0/eval97.pem.ctm.filt.sys

# Audio-transcript alignment stage 1
%WER 19.8 | 728 32834 | 82.3 12.7 5.1 2.1 19.8 88.0 | exp/tri4_b/decode_nosp_eval97.pem_rescore/score_14_0.5/eval97.pem.ctm.filt.sys
%WER 20.9 | 728 32834 | 81.2 13.4 5.4 2.1 20.9 88.7 | exp/tri4_b/decode_nosp_eval97.pem/score_14_0.0/eval97.pem.ctm.filt.sys

# Audio-transcript alignment stage 2
%WER 19.9 | 728 32834 | 82.3 13.2 4.5 2.3 19.9 88.9 | exp/tri4_2b/decode_nosp_eval97.pem_rescore/score_13_0.0/eval97.pem.ctm.filt.sys
%WER 21.2 | 728 32834 | 81.3 14.3 4.4 2.5 21.2 89.8 | exp/tri4_2b/decode_nosp_eval97.pem/score_12_0.0/eval97.pem.ctm.filt.sys

# Cleaned transcripts
%WER 19.0 | 728 32834 | 83.1 12.5 4.4 2.1 19.0 87.0 | exp/tri5_2b_cleaned/decode_nosp_eval97.pem_rescore/score_14_0.0/eval97.pem.ctm.filt.sys
%WER 20.2 | 728 32834 | 82.1 13.4 4.5 2.3 20.2 89.0 | exp/tri5_2b_cleaned/decode_nosp_eval97.pem/score_13_0.0/eval97.pem.ctm.filt.sys

# Oracle transcripts
%WER 18.0 | 728 32834 | 83.9 11.7 4.3 2.0 18.0 85.9 | exp/tri4/decode_nosp_eval97.pem_rescore/score_14_0.0/eval97.pem.ctm.filt.sys
%WER 19.3 | 728 32834 | 82.9 12.6 4.6 2.2 19.3 86.8 | exp/tri4/decode_nosp_eval97.pem/score_13_0.0/eval97.pem.ctm.filt.sys

. ./cmd.sh
. ./path.sh

segment_stage=-8
nj=40
reco_nj=80
affix=b
new_affix=2b

. utils/parse_options.sh

###############################################################################
## Simulate unsegmented data directory.
###############################################################################
utils/data/convert_data_dir_to_whole.sh data/train data/train_long

steps/make_mfcc.sh --cmd "$train_cmd --max-jobs-run 40" --nj $reco_nj \
  data/train_long exp/make_mfcc/train_long mfcc || exit 1
steps/compute_cmvn_stats.sh data/train_long \
  exp/make_mfcc/train_long mfcc
utils/fix_data_dir.sh data/train_long

###############################################################################
## Train WSJ models.
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
# Segment long recordings using TF-IDF retrieval of reference text 
# for uniformly segmented audio chunks based on modified Levenshtein alignment.
# Use a SAT model trained on train_si284 (wsj_tri3)
###############################################################################

steps/cleanup/segment_long_utterances.sh --cmd "$train_cmd" \
  --stage $segment_stage --nj $reco_nj \
  --max-bad-proportion 0.5 --align-full-hyp true \
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
# Segment long recordings using TF-IDF retrieval of reference text 
# for uniformly segmented audio chunks based on modified Levenshtein alignment.
# Use a SAT model trained on tri4_a
###############################################################################

steps/cleanup/segment_long_utterances.sh --cmd "$train_cmd" \
  --stage $segment_stage --nj $reco_nj \
  --max-bad-proportion 0.5 --align-full-hyp true \
  exp/tri4_${affix} data/lang_nosp data/train_long data/train_long/text \
  data/train_reseg_${new_affix} exp/segment_long_utts_${new_affix}_train

steps/compute_cmvn_stats.sh data/train_reseg_${new_affix} \
  exp/make_mfcc/train_reseg_${new_affix} mfcc
utils/fix_data_dir.sh data/train_reseg_${new_affix}

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
