#!/bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0

# This script demonstrates how to re-segment training data selecting only the 
# "good" audio that matches the transcripts.
# The basic idea is to decode with an existing in-domain acoustic model, and a
# biased language model built from the reference, and then work out the
# segmentation from a ctm like file.

# For nnet3 and chain results after cleanup, see the scripts in 
# local/nnet3/run_tdnn.sh and local/chain/run_tdnn.sh 

# GMM Results for speaker-independent (SI) and speaker adaptive training (SAT) systems on dev and test sets

# SI systems:

# Baseline
## %WER 26.7 | 507 17792 | 77.7 16.9 5.4 4.5 26.7 95.3 | -0.329 | exp/tri3/decode_dev.si/score_12_1.0/ctm.filt.filt.sys
## %WER 25.5 | 1155 27512 | 77.9 17.4 4.6 3.4 25.5 92.9 | -0.270 | exp/tri3/decode_test.si/score_12_1.0/ctm.filt.filt.sys
# With cleanup 
## %WER 26.2 | 507 17792 | 77.9 16.3 5.8 4.2 26.2 95.7 | -0.340 | exp/tri3_cleaned_b/decode_dev.si/score_13_1.0/ctm.filt.filt.sys
## %WER 25.2 | 1155 27512 | 78.0 17.1 4.9 3.2 25.2 92.4 | -0.257 | exp/tri3_cleaned_b/decode_test.si/score_13_1.0/ctm.filt.filt.sys

# SAT systems:

# Baseline
## %WER 22.0 | 507 17792 | 81.6 13.2 5.2 3.6 22.0 93.9 | -0.189 | exp/tri3/decode_dev/score_13_1.0/ctm.filt.filt.sys
## %WER 20.3 | 1155 27512 | 82.7 13.4 3.9 3.0 20.3 90.0 | -0.063 | exp/tri3/decode_test/score_14_0.5/ctm.filt.filt.sys
# With cleanup 
## %WER 21.9 | 507 17792 | 81.6 13.0 5.4 3.5 21.9 93.7 | -0.147 | exp/tri3_cleaned_b/decode_dev/score_13_1.0/ctm.filt.filt.sys
## %WER 19.8 | 1155 27512 | 83.0 13.0 4.0 2.8 19.8 89.5 | -0.041 | exp/tri3_cleaned_b/decode_test/score_14_0.5/ctm.filt.filt.sys

set -e 
set -o pipefail 
set -u

stage=0
cleanup_stage=0

pad_length=5                  # Number of frames for padding the created 
                              # subsegments
max_silence_length=50         # Maxium number of silence frames above which they are removed and the segment is split
silence_padding_correct=5     # The amount of silence frames to pad a segment by 
                              # if the silence is next to a correct hypothesis word
silence_padding_incorrect=20  # The amount of silence frames to pad a segment by 
                              # if the silence is next to an incorrect hypothesis word
min_wer_for_splitting=10      # Minimum WER% for a segment to be considered for splitting

cleanup_affix=cleaned
nj=100

. ./path.sh
. ./cmd.sh
. utils/parse_options.sh 

ngram_order=2
top_n_words=100

gmm_dir=exp/tri3

bad_utts_dir=${gmm_dir}_train_split_bad_utts 

lm_affix="o$ngram_order"
if [ $ngram_order -eq 1 ]; then
  lm_affix=${lm_affix}_top$top_n_words
fi

bad_utts_dir=${bad_utts_dir}_${lm_affix}

if [ $stage -le 1 ]; then
  steps/cleanup/do_cleanup_segmentation.sh \
    --cmd "$train_cmd" --nj $nj \
    --pad-length $pad_length \
    --silence-padding-correct $silence_padding_correct \
    --silence-padding-incorrect $silence_padding_incorrect \
    --max-silence-length $max_silence_length \
    --max-incorrect-words 0 \
    --min-correct-frames 0 \
    --max-utterance-wer 20000 \
    --min-wer-for-splitting $min_wer_for_splitting \
    --stage $cleanup_stage \
    data/train data/lang $gmm_dir \
    $bad_utts_dir $bad_utts_dir/segmentation_${cleanup_affix} \
    data/train_${cleanup_affix}
fi

if [ $stage -le 2 ]; then
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj \
    data/train_${cleanup_affix} exp/make_mfcc mfcc
  steps/compute_cmvn_stats.sh \
    data/train_${cleanup_affix} exp/make_mfcc mfcc
  utils/fix_data_dir.sh data/train_${cleanup_affix}
fi

nj=50

if [ $stage -le 3 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train_${cleanup_affix} data/lang exp/tri2 exp/tri2_ali_${cleanup_affix}
fi

if [ $stage -le 4 ]; then
  steps/train_sat.sh --cmd "$train_cmd" \
    5000 100000 data/train_${cleanup_affix} data/lang \
    exp/tri2_ali_${cleanup_affix} exp/tri3_${cleanup_affix}
fi

nj_dev=$(cat data/dev/spk2utt | wc -l)
nj_test=$(cat data/test/spk2utt | wc -l)

if [ $stage -le 5 ]; then
  graph_dir=exp/tri3_${cleanup_affix}/graph
  $decode_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_test exp/tri3_${cleanup_affix} $graph_dir

  steps/decode_fmllr.sh --nj $nj_dev --cmd "$decode_cmd" \
    $graph_dir data/dev exp/tri3_${cleanup_affix}/decode_dev
  steps/decode_fmllr.sh --nj $nj_test --cmd "$decode_cmd" \
    $graph_dir data/test exp/tri3_${cleanup_affix}/decode_test
fi
