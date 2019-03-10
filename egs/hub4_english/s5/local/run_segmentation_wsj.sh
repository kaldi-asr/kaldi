#!/bin/bash

# Copyright 2016-18  Vimal Manohar
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

# In step 3, if you need to align the full hypothesis of audio with the 
# reference text as opposed to finding the best matching substring, 
# then use --align-full-hyp true in the scripts below.

# WSJ models (From step 1)
# %WER 29.9 | 728 32834 | 72.9 17.8 9.3 2.8 29.9 92.7 | exp/wsj_tri3/decode_nosp_eval97.pem_rescore/score_16_0.0/eval97.pem.ctm.filt.sys
# %WER 30.8 | 728 32834 | 71.8 18.4 9.8 2.6 30.8 92.3 | exp/wsj_tri3/decode_nosp_eval97.pem/score_17_0.0/eval97.pem.ctm.filt.sys

# In-domain GMM (From step 4) -- 107 hrs
# %WER 19.1 | 728 32834 | 82.7 12.2 5.1 1.9 19.1 86.4 | exp/tri4_a/decode_nosp_eval97.pem_rescore/score_14_1.0/eval97.pem.ctm.filt.sys
# %WER 20.4 | 728 32834 | 81.6 13.1 5.3 2.1 20.4 87.4 | exp/tri4_a/decode_nosp_eval97.pem/score_14_0.0/eval97.pem.ctm.filt.sys

# Stage 2 in-domain GMM (From step 5) -- 124 hrs
# %WER 20.9 | 728 32834 | 81.2 13.6 5.3 2.1 20.9 87.4 | exp/tri4_2a/decode_nosp_eval97.pem/score_14_0.0/eval97.pem.ctm.filt.sys
# %WER 19.8 | 728 32834 | 82.3 12.9 4.7 2.2 19.8 86.1 | exp/tri4_2a/decode_nosp_eval97.pem_rescore/score_12_0.5/eval97.pem.ctm.filt.sys

# GMM trained on cleaned transcripts (From step 6) -- 120 hrs
# %WER 18.4 | 728 32834 | 83.6 11.9 4.5 2.1 18.4 84.8 | exp/tri5_2a_cleaned/decode_nosp_eval97.pem_rescore/score_13_0.0/eval97.pem.ctm.filt.sys
# %WER 19.6 | 728 32834 | 82.5 12.7 4.8 2.2 19.6 86.8 | exp/tri5_2a_cleaned/decode_nosp_eval97.pem/score_13_0.0/eval97.pem.ctm.filt.sys

# Oracle HUB4 transcripts -- 148 hrs
# %WER 17.8 | 728 32834 | 84.1 11.8 4.1 1.9 17.8 82.8 | exp/tri4/decode_nosp_eval97.pem_rescore/score_13_0.5/eval97.pem.ctm.filt.sys
# %WER 19.0 | 728 32834 | 83.0 12.7 4.3 2.0 19.0 84.2 | exp/tri4/decode_nosp_eval97.pem/score_13_0.0/eval97.pem.ctm.filt.sys

stage=0
segment_stage=-8
nj=40
reco_nj=80
stage1_affix=a    # For steps 2, 3 and 4 above
stage2_affix=2a   # For step 5 above

# WSJ run.sh must be run until the data preparation stage
wsj_base=../../wsj/s5   # Change this to the WSJ base directory

if [ -f ./path.sh ]; then . ./path.sh; fi
. ./cmd.sh

. utils/parse_options.sh

if [ ! -f $wsj_base/data/train_si284/wav.scp ]; then
  echo "WSJ data directory $wsj_base/data/train_si284 is not prepared."
  echo "Run the initial stages of WSJ's run.sh"
  exit 0
fi

if [ $stage -le 0 ]; then
  # We copy the prepared data to the current directory
  utils/copy_data_dir.sh $wsj_base/data/train_si84_2kshort data/train_si84_2kshort
  utils/copy_data_dir.sh $wsj_base/data/train_si84 data/train_si84
  utils/copy_data_dir.sh $wsj_base/data/train_si284 data/train_si284
fi

###############################################################################
## Simulate unsegmented HUB4 data directory.
###############################################################################

if [ $stage -le 1 ]; then
  utils/data/convert_data_dir_to_whole.sh data/train data/train_long

  steps/make_mfcc.sh --cmd "$train_cmd --max-jobs-run 40" \
    --nj $reco_nj --write-utt2num-frames true \
    data/train_long exp/make_mfcc/train_long mfcc
  steps/compute_cmvn_stats.sh data/train_long \
    exp/make_mfcc/train_long mfcc
  utils/fix_data_dir.sh data/train_long
fi

###############################################################################
## Train GMM on out-of-domain WSJ corpus 
###############################################################################

if [ $stage -le 2 ]; then
  steps/train_mono.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
    data/train_si84_2kshort data/lang_nosp exp/wsj_mono0a
fi

if [ $stage -le 3 ]; then
  steps/align_si.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
    data/train_si84 data/lang_nosp exp/wsj_mono0a exp/wsj_mono0a_ali_si84

  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2500 15000 \
    data/train_si84 data/lang_nosp exp/wsj_mono0a_ali_si84 exp/wsj_tri1
fi

if [ $stage -le 4 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train_si284 data/lang_nosp exp/wsj_tri1 exp/wsj_tri1_ali_si284

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 4000 42000 \
    data/train_si284 data/lang_nosp exp/wsj_tri1_ali_si284 exp/wsj_tri2
fi

if [ $stage -le 5 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train_si284 data/lang_nosp exp/wsj_tri2 exp/wsj_tri2_ali_si284

  steps/train_sat.sh --cmd "$train_cmd" \
    4000 42000 \
    data/train_si284 data/lang_nosp exp/wsj_tri2_ali_si284 exp/wsj_tri3
fi

if [ $stage -le 6 ]; then
  utils/mkgraph.sh data/lang_nosp_test \
    exp/wsj_tri3/{,graph_nosp_test}

  for dset in eval97.pem; do
    this_nj=`cat data/$dset/spk2utt | wc -l`
    if [ $this_nj -gt 20 ]; then
      this_nj=20
    fi
    steps/decode_fmllr.sh --nj $this_nj --cmd "$decode_cmd" --num-threads 4 \
      exp/wsj_tri3/graph_nosp_test data/$dset \
      exp/wsj_tri3/decode_nosp_${dset}
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_nosp_test data/lang_nosp_test_rescore \
      data/${dset} exp/wsj_tri3/decode_nosp_${dset} \
      exp/wsj_tri3/decode_nosp_${dset}_rescore
  done
fi

###############################################################################
# Segment long HUB4 recordings and retrieve transcript using 
# Smith-Waterman alignment.
# Use a SAT model trained on train_si284 (wsj_tri3) as seed model for decoding.
###############################################################################

if [ $stage -le 7 ]; then
  steps/cleanup/segment_long_utterances.sh --cmd "$train_cmd" \
    --stage $segment_stage --nj $reco_nj \
    --max-bad-proportion 0.5 --align-full-hyp false \
    exp/wsj_tri3 data/lang_nosp data/train_long \
    data/train_reseg_${stage1_affix} exp/segment_long_utts_${stage1_affix}_train
fi

if [ $stage -le 8 ]; then
  steps/compute_cmvn_stats.sh data/train_reseg_${stage1_affix} \
    exp/make_mfcc/train_reseg_${stage1_affix} mfcc
  utils/fix_data_dir.sh data/train_reseg_${stage1_affix}

  utils/data/modify_speaker_info.sh data/train_reseg_${stage1_affix} \
    data/train_reseg_${stage1_affix}_spk30sec
  steps/compute_cmvn_stats.sh data/train_reseg_${stage1_affix}_spk30sec \
    exp/make_mfcc/train_reseg_${stage1_affix}_spk30sec mfcc
  utils/fix_data_dir.sh data/train_reseg_${stage1_affix}_spk30sec
fi

###############################################################################
# Train new in-domain GMM (tri4_a) on retrieved transcripts.
###############################################################################

if [ $stage -le 9 ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/train_reseg_${stage1_affix}_spk30sec data/lang_nosp \
    exp/wsj_tri3 exp/wsj_tri3_ali_train_reseg_${stage1_affix}

  steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
    data/train_reseg_${stage1_affix}_spk30sec data/lang_nosp \
    exp/wsj_tri3_ali_train_reseg_${stage1_affix} exp/tri3_${stage1_affix} 
fi

if [ $stage -le 10 ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/train_reseg_${stage1_affix}_spk30sec data/lang_nosp exp/tri3_${stage1_affix} exp/tri3_${stage1_affix}_ali

  steps/train_sat.sh --cmd "$train_cmd" 5000 100000 \
    data/train_reseg_${stage1_affix}_spk30sec data/lang_nosp exp/tri3_${stage1_affix}_ali exp/tri4_${stage1_affix}
fi

if [ $stage -le 11 ]; then
  utils/mkgraph.sh data/lang_nosp_test exp/tri4_${stage1_affix}/{,graph_nosp_test}
  for dset in eval97.pem; do
    this_nj=`cat data/$dset/spk2utt | wc -l`
    if [ $this_nj -gt 20 ]; then
      this_nj=20
    fi
    steps/decode_fmllr.sh --nj $this_nj --cmd "$decode_cmd" --num-threads 4 \
      exp/tri4_${stage1_affix}/graph_nosp_test data/$dset exp/tri4_${stage1_affix}/decode_nosp_${dset}
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_nosp_test data/lang_nosp_test_rescore \
      data/${dset} exp/tri4_${stage1_affix}/decode_nosp_${dset} \
      exp/tri4_${stage1_affix}/decode_nosp_${dset}_rescore
  done
fi

###############################################################################
# Segment long HUB4 recordings and retrieve transcript using 
# Smith-Waterman alignment.
# Use in-domain SAT model (tri4_a) as seed model for decoding.
###############################################################################

if [ $stage -le 12 ]; then
  steps/cleanup/segment_long_utterances.sh --cmd "$train_cmd" \
    --stage $segment_stage --nj $reco_nj \
    --max-bad-proportion 0.5 --align-full-hyp false \
    exp/tri4_${stage1_affix} data/lang_nosp data/train_long \
    data/train_reseg_${stage2_affix} exp/segment_long_utts_${stage2_affix}_train
fi

if [ $stage -le 13 ]; then
  steps/compute_cmvn_stats.sh data/train_reseg_${stage2_affix} \
    exp/make_mfcc/train_reseg_${stage2_affix} mfcc
  utils/fix_data_dir.sh data/train_reseg_${stage2_affix}
fi

###############################################################################
# Train new in-domain GMM (tri4_2a) on retrieved transcripts.
###############################################################################

if [ $stage -le 14 ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/train_reseg_${stage2_affix} data/lang_nosp \
    exp/tri4_${stage1_affix} exp/tri4_${stage1_affix}_ali_train_reseg_${stage2_affix}

  steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
    data/train_reseg_${stage2_affix} data/lang_nosp \
    exp/tri4_${stage1_affix}_ali_train_reseg_${stage2_affix} exp/tri4_${stage2_affix} 
fi

if [ $stage -le 15 ]; then
  utils/mkgraph.sh data/lang_nosp_test exp/tri4_${stage2_affix}/{,graph_nosp_test}
  for dset in eval97.pem; do
    this_nj=`cat data/$dset/spk2utt | wc -l`
    if [ $this_nj -gt 20 ]; then
      this_nj=20
    fi
    steps/decode_fmllr.sh --nj $this_nj --cmd "$decode_cmd" --num-threads 4 \
      exp/tri4_${stage2_affix}/graph_nosp_test data/$dset exp/tri4_${stage2_affix}/decode_nosp_${dset}
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_nosp_test data/lang_nosp_test_rescore \
      data/${dset} exp/tri4_${stage2_affix}/decode_nosp_${dset} \
      exp/tri4_${stage2_affix}/decode_nosp_${dset}_rescore
  done
fi

###############################################################################
# Cleanup transcripts
# Use in-domain SAT model (tri4_2a) as seed model for decoding.
###############################################################################

cleanup_stage=-1
cleanup_affix=cleaned
srcdir=exp/tri4_${stage2_affix}
cleaned_data=data/train_reseg_${stage2_affix}_${cleanup_affix}
dir=${srcdir}_${cleanup_affix}_work
cleaned_dir=${srcdir}_${cleanup_affix}

if [ $stage -le 16 ]; then
  steps/cleanup/clean_and_segment_data.sh --stage $cleanup_stage --nj 80 \
    --cmd "$train_cmd" \
    data/train_reseg_${stage2_affix} data/lang_nosp \
    $srcdir $dir $cleaned_data
fi

###############################################################################
# Train new in-domain GMM (tri4_2a) on cleaned-up transcripts.
###############################################################################

if [ $stage -le 17 ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    $cleaned_data data/lang_nosp $srcdir ${srcdir}_ali_${cleanup_affix}

  steps/train_sat.sh --cmd "$train_cmd" \
    5000 100000 $cleaned_data data/lang_nosp \
    ${srcdir}_ali_${cleanup_affix} exp/tri5_${stage2_affix}_${cleanup_affix}
fi

if [ $stage -le 18 ]; then
  utils/mkgraph.sh data/lang_nosp_test \
    exp/tri5_${stage2_affix}_${cleanup_affix}/{,graph_nosp_test}
  for dset in eval97.pem; do
    this_nj=`cat data/$dset/spk2utt | wc -l`
    if [ $this_nj -gt 20 ]; then
      this_nj=20
    fi
    steps/decode_fmllr.sh --nj $this_nj --cmd "$decode_cmd" --num-threads 4 \
      exp/tri5_${stage2_affix}_${cleanup_affix}/graph_nosp_test data/$dset \
      exp/tri5_${stage2_affix}_${cleanup_affix}/decode_nosp_${dset}
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_nosp_test data/lang_nosp_test_rescore \
      data/${dset} exp/tri5_${stage2_affix}_${cleanup_affix}/decode_nosp_${dset} \
      exp/tri5_${stage2_affix}_${cleanup_affix}/decode_nosp_${dset}_rescore
  done
fi

exit 0
