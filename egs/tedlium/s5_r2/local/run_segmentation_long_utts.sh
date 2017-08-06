#!/bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0

# This script demonstrates how to re-segment long audios into short segments.
# The basic idea is to decode with an existing in-domain acoustic model, and a
# bigram language model built from the reference, and then work out the
# segmentation from a ctm like file.

## %WER results. 

## Baseline results
# %WER 18.1 | 507 17783 | 84.7 10.7 4.6 2.8 18.1 91.1 | -0.073 | exp/tri3/decode_dev_rescore/score_16_0.0/ctm.filt.filt.sys
# %WER 16.6 | 1155 27500 | 85.7 10.7 3.6 2.4 16.6 86.0 | -0.041 | exp/tri3/decode_test_rescore/score_16_0.0/ctm.filt.filt.sys

## With Cleanup
# %WER 18.0 | 507 17783 | 85.0 10.6 4.4 3.0 18.0 90.9 | -0.064 | exp/tri3_cleaned/decode_dev_rescore/score_14_0.0/ctm.filt.filt.sys
# %WER 16.6 | 1155 27500 | 85.9 10.8 3.3 2.5 16.6 86.6 | -0.050 | exp/tri3_cleaned/decode_test_rescore/score_14_0.0/ctm.filt.filt.sys

## Segmentation results
# %WER 18.9 | 507 17783 | 83.9 11.1 5.0 2.8 18.9 92.9 | -0.103 | exp/tri3_reseg_a/decode_nosp_dev_rescore/score_14_0.0/ctm.filt.filt.sys
# %WER 17.6 | 1155 27500 | 84.6 11.3 4.1 2.2 17.6 86.8 | -0.005 | exp/tri3_reseg_a/decode_nosp_test_rescore/score_14_0.0/ctm.filt.filt.sys

## Segmentation + Cleanup

# cleaned - 
# Default segmentation-opts "--max-junk-proportion=1 --max-deleted-words-kept-when-merging=1 --min-split-point-duration=0.1" 
# cleaned_b -
# "--max-junk-proportion=0.5 --max-deleted-words-kept-when-merging=10"
# cleaned_c -
# "--max-junk-proportion=0.2 --max-deleted-words-kept-when-merging=6 --min-split-point-duration=0.3"

# %WER 18.7 | 507 17783 | 84.0 11.0 5.0 2.8 18.7 91.7 | -0.119 | exp/tri3_reseg_a_cleaned/decode_nosp_dev_rescore/score_15_0.0/ctm.filt.filt.sys
# %WER 18.6 | 507 17783 | 84.0 11.0 4.9 2.7 18.6 91.5 | -0.092 | exp/tri3_reseg_a_cleaned_b/decode_nosp_dev_rescore/score_15_0.0/ctm.filt.filt.sys
# %WER 18.6 | 507 17783 | 84.1 10.8 5.0 2.7 18.6 92.1 | -0.114 | exp/tri3_reseg_a_cleaned_c/decode_nosp_dev_rescore/score_15_0.0/ctm.filt.filt.sys

# %WER 17.7 | 1155 27500 | 84.5 11.4 4.0 2.2 17.7 86.8 | -0.020 | exp/tri3_reseg_a_cleaned/decode_nosp_test_rescore/score_14_0.0/ctm.filt.filt.sys
# %WER 17.3 | 1155 27500 | 84.8 11.2 4.1 2.1 17.3 86.8 | -0.002 | exp/tri3_reseg_a_cleaned_b/decode_nosp_test_rescore/score_15_0.0/ctm.filt.filt.sys
# %WER 17.7 | 1155 27500 | 84.6 11.4 4.1 2.3 17.7 86.6 | -0.018 | exp/tri3_reseg_a_cleaned_c/decode_nosp_test_rescore/score_14_0.0/ctm.filt.filt.sys

## Use silence and pronunciation probs estimated from resegmented data
# %WER 18.2 | 507 17783 | 84.6 10.8 4.5 2.9 18.2 92.5 | -0.037 | exp/tri3_reseg_a/decode_a_dev_rescore/score_16_0.0/ctm.filt.filt.sys
# %WER 16.9 | 1155 27500 | 85.5 11.0 3.5 2.4 16.9 86.1 | -0.024 | exp/tri3_reseg_a/decode_a_test_rescore/score_14_0.0/ctm.filt.filt.sys

## Use silence and pronunciation probs estimated from resegmented and cleaned up data
# %WER 18.2 | 507 17783 | 84.4 10.8 4.9 2.6 18.2 92.5 | -0.074 | exp/tri3_reseg_a_cleaned_b/decode_a_cleaned_b_dev_rescore/score_15_0.5/ctm.filt.filt.sys
# %WER 16.8 | 1155 27500 | 85.4 10.7 3.9 2.1 16.8 86.8 | -0.046 | exp/tri3_reseg_a_cleaned_b/decode_a_cleaned_b_test_rescore/score_14_0.5/ctm.filt.filt.sys

. ./cmd.sh
. ./path.sh

set -e -o pipefail -u

segment_stage=-9
cleanup_stage=-1
cleanup_affix=cleaned_b
affix=_a

decode_nj=8   # note: should not be >38 which is the number of speakers in the dev set
              # after applying --seconds-per-spk-max 180.  We decode with 4 threads, so
              # this will be too many jobs if you're using run.pl.

###############################################################################
# Simulate unsegmented data directory.
###############################################################################
utils/data/convert_data_dir_to_whole.sh data/train data/train_long

###############################################################################
# Train system on a small subset of 2000 utterances that are 
# manually segmented.
###############################################################################

utils/subset_data_dir.sh --speakers data/train 2000 data/train_2k
utils/subset_data_dir.sh --shortest data/train_2k 500 data/train_2k_short500

steps/make_mfcc.sh --cmd "$train_cmd" --nj 32 \
  data/train_long exp/make_mfcc/train_long mfcc || exit 1
steps/compute_cmvn_stats.sh data/train_long \
  exp/make_mfcc/train_long mfcc


steps/train_mono.sh --nj 20 --cmd "$train_cmd" \
  data/train_2k_short500 data/lang_nosp exp/mono_a

steps/align_si.sh --nj 20 --cmd "$train_cmd" \
  data/train_2k data/lang_nosp exp/mono_a exp/mono_a_ali_2k

steps/train_deltas.sh --cmd "$train_cmd" \
  500 5000 data/train_2k data/lang_nosp exp/mono_a_ali_2k exp/tri1a

steps/align_si.sh --nj 20 --cmd "$train_cmd" \
  data/train_2k data/lang_nosp exp/tri1a exp/tri1a_ali

steps/train_lda_mllt.sh --cmd "$train_cmd" \
  1000 10000 data/train_2k data/lang_nosp exp/tri1a_ali exp/tri1b

###############################################################################
# Segment long recordings using TF-IDF retrieval of reference text 
# for uniformly segmented audio chunks based on Smith-Waterman alignment.
# Use a model trained on train_2k (tri1b)
###############################################################################

steps/cleanup/segment_long_utterances.sh --cmd "$train_cmd" \
  --stage $segment_stage --nj 80 \
  --max-bad-proportion 0.5 \
  exp/tri1b data/lang_nosp data/train_long data/train_reseg${affix} \
  exp/segment_long_utts${affix}_train

steps/compute_cmvn_stats.sh data/train_reseg${affix} \
  exp/make_mfcc/train_reseg${affix} mfcc
utils/fix_data_dir.sh data/train_reseg${affix}

###############################################################################
# Train new model on segmented data directory starting from the same model
# used for segmentation. (tri2_reseg)
###############################################################################

# Align tri1b system with reseg${affix} data
steps/align_si.sh  --nj 40 --cmd "$train_cmd" \
  data/train_reseg${affix} \
  data/lang_nosp exp/tri1b exp/tri1b_ali_reseg${affix}  || exit 1;

# Train LDA+MLLT system on reseg${affix} data
steps/train_lda_mllt.sh --cmd "$train_cmd" \
  4000 50000 data/train_reseg${affix} data/lang_nosp \
  exp/tri1b_ali_reseg${affix} exp/tri2_reseg${affix}

(
utils/mkgraph.sh data/lang_nosp exp/tri2_reseg${affix} \
  exp/tri2_reseg${affix}/graph_nosp
for dset in dev test; do
  steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
    exp/tri2_reseg${affix}/graph_nosp data/${dset} \
    exp/tri2_reseg${affix}/decode_nosp_${dset}
  steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" data/lang_nosp \
    data/lang_nosp_rescore \
     data/${dset} exp/tri2_reseg${affix}/decode_nosp_${dset} \
     exp/tri2_reseg${affix}/decode_nosp_${dset}_rescore
done
) &

###############################################################################
# Train SAT model on segmented data directory
###############################################################################

# Train SAT system on reseg${affix} data
steps/train_sat.sh --cmd "$train_cmd" 5000 100000 \
  data/train_reseg${affix} data/lang_nosp \
  exp/tri2_reseg${affix} exp/tri3_reseg${affix}

(
utils/mkgraph.sh data/lang_nosp exp/tri3_reseg${affix} \
  exp/tri3_reseg${affix}/graph_nosp
for dset in dev test; do
  steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
    exp/tri3_reseg${affix}/graph_nosp data/${dset} \
    exp/tri3_reseg${affix}/decode_nosp_${dset}
  steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" data/lang_nosp \
    data/lang_nosp_rescore \
     data/${dset} exp/tri3_reseg${affix}/decode_nosp_${dset} \
     exp/tri3_reseg${affix}/decode_nosp_${dset}_rescore
done
) &

###############################################################################
# Clean and segmented data
###############################################################################

segmentation_opts=(
--max-junk-proportion=0.5
--max-deleted-words-kept-when-merging=10
)
opts="${segmentation_opts[@]}"

steps/cleanup/clean_and_segment_data.sh --nj 40 --cmd "$train_cmd" \
  --segmentation-opts "$opts" \
  data/train_reseg${affix} data/lang_nosp exp/tri3_reseg${affix} \
  exp/tri3_reseg${affix}_${cleanup_affix}_work \
  data/train_reseg${affix}_${cleanup_affix}

###############################################################################
# Train new SAT model on cleaned data directory 
###############################################################################

steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
  data/train_reseg${affix}_${cleanup_affix} data/lang_nosp \
  exp/tri3_reseg${affix} exp/tri3_reseg${affix}_ali_${cleanup_affix}

steps/train_sat.sh --cmd "$train_cmd" \
  5000 100000 data/train_reseg${affix}_${cleanup_affix} data/lang_nosp \
  exp/tri3_reseg${affix}_ali_${cleanup_affix} \
  exp/tri3_reseg${affix}_$cleanup_affix

(
utils/mkgraph.sh data/lang_nosp exp/tri3_reseg${affix}_$cleanup_affix \
  exp/tri3_reseg${affix}_$cleanup_affix/graph_nosp
for dset in dev test; do
  steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
    exp/tri3_reseg${affix}_$cleanup_affix/graph_nosp data/${dset} \
    exp/tri3_reseg${affix}_$cleanup_affix/decode_nosp_${dset}
  steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" data/lang_nosp \
    data/lang_nosp_rescore \
     data/${dset} exp/tri3_reseg${affix}_$cleanup_affix/decode_nosp_${dset} \
     exp/tri3_reseg${affix}_$cleanup_affix/decode_nosp_${dset}_rescore
done
) &

steps/get_prons.sh --cmd "$train_cmd" \
  data/train_reseg${affix}_${cleanup_affix} \
  data/lang_nosp exp/tri3_reseg${affix}_$cleanup_affix
utils/dict_dir_add_pronprobs.sh --max-normalize true \
  data/local/dict_nosp \
  exp/tri3_reseg${affix}_$cleanup_affix/{pron,sil,pron_bigram}_counts_nowb.txt \
  data/local/dict${affix}_$cleanup_affix

utils/prepare_lang.sh data/local/dict${affix}_$cleanup_affix \
  "<unk>" data/local/lang data/lang${affix}_$cleanup_affix
cp -rT data/lang${affix}_$cleanup_affix data/lang${affix}_${cleanup_affix}_rescore
cp data/lang_nosp/G.fst data/lang${affix}_$cleanup_affix/
cp data/lang_nosp_rescore/G.carpa data/lang${affix}_${cleanup_affix}_rescore/

(
utils/mkgraph.sh data/lang${affix}_${cleanup_affix} \
  exp/tri3_reseg${affix}_$cleanup_affix{,/graph${affix}_${cleanup_affix}} 

for dset in dev test; do
  steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
    exp/tri3_reseg${affix}_$cleanup_affix/graph${affix}_${cleanup_affix} \
    data/${dset} \
    exp/tri3_reseg${affix}_$cleanup_affix/decode${affix}_${cleanup_affix}_${dset}
  steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" data/lang${affix}_${cleanup_affix} \
    data/lang${affix}_${cleanup_affix}_rescore \
     data/${dset} exp/tri3_reseg${affix}_$cleanup_affix/decode${affix}_${cleanup_affix}_${dset} \
     exp/tri3_reseg${affix}_$cleanup_affix/decode${affix}_${cleanup_affix}_${dset}_rescore
done
) &

steps/get_prons.sh --cmd "$train_cmd" \
  data/train_reseg${affix} \
  data/lang_nosp exp/tri3_reseg${affix}
utils/dict_dir_add_pronprobs.sh --max-normalize true \
  data/local/dict_nosp \
  exp/tri3_reseg${affix}/{pron,sil,pron_bigram}_counts_nowb.txt \
  data/local/dict${affix}

utils/prepare_lang.sh data/local/dict${affix} \
  "<unk>" data/local/lang data/lang${affix}
cp -rT data/lang${affix} data/lang${affix}_rescore
cp data/lang_nosp/G.fst data/lang${affix}/
cp data/lang_nosp_rescore/G.carpa data/lang${affix}_rescore/

(
utils/mkgraph.sh data/lang${affix} \
  exp/tri3_reseg${affix}{,/graph${affix}} 

for dset in dev test; do
  steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
    exp/tri3_reseg${affix}/graph${affix} \
    data/${dset} \
    exp/tri3_reseg${affix}/decode${affix}_${dset}
  steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" data/lang${affix} \
    data/lang${affix}_rescore \
     data/${dset} exp/tri3_reseg${affix}/decode${affix}_${dset} \
     exp/tri3_reseg${affix}/decode${affix}_${dset}_rescore
done
) &

wait
exit 0
