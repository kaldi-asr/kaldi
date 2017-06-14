#! /bin/bash

. ./cmd.sh
. ./path.sh

set -e -o pipefail -u

# This differs from _d by using the --align-full-hyp false, which
# gets best matching subsequence of reference and hypothesis using
# Smith-Waterman alignment,
# as against using Levenshtein distance w.r.t. full hypothesis.

# _d
# STAGE 2 | %WER 19.1 | 507 17783 | 84.1 10.7 5.2 3.2 19.1 91.1 | -0.193 | exp/tri5_2d_cleaned/decode_nosp_dev_rescore/score_14_0.0/ctm.filt.filt.sys
# STAGE 2 | %WER 17.7 | 1155 27500 | 84.6 11.4 3.9 2.3 17.7 87.4 | -0.076 | exp/tri5_2d_cleaned/decode_nosp_test_rescore/score_13_0.0/ctm.filt.filt.sys

# _e
# STAGE 3 | %WER 17.4 | 1155 27500 | 85.0 11.4 3.7 2.4 17.4 87.2 | -0.086 | exp/tri5_2e_cleaned/decode_nosp_test_rescore/score_12_0.0/ctm.filt.filt.sys
# STAGE 3 | %WER 18.8 | 507 17783 | 84.4 10.7 4.9 3.2 18.8 91.3 | -0.162 | exp/tri5_2e_cleaned/decode_nosp_dev_rescore/score_14_0.0/ctm.filt.filt.sys

# Note: Better results can be obtained by using silence and pronunciation
# probs as seen in STAGE 2.

segment_stage=-10
affix=_1e
decode_nj=30
cleanup_stage=-10

###############################################################################
# Segment long recordings using TF-IDF retrieval of reference text 
# for uniformly segmented audio chunks based on Smith-Waterman alignment.
# Use a model trained on WSJ train_si84 (tri2b)
###############################################################################

###
# STAGE 0
###

utils/data/convert_data_dir_to_whole.sh data/train data/train_long
steps/make_mfcc.sh --nj 40 --cmd "$train_cmd" \
  data/train_long exp/make_mfcc/train_long mfcc
steps/compute_cmvn_stats.sh \
  data/train_long exp/make_mfcc/train_long mfcc
utils/fix_data_dir.sh data/train_long

steps/cleanup/segment_long_utterances.sh \
  --cmd "$train_cmd" --nj 80 \
  --stage $segment_stage \
  --max-bad-proportion 0.5 --align-full-hyp false \
  exp/wsj_tri2b data/lang_nosp data/train_long data/train_reseg${affix} \
  exp/segment_wsj_long_utts${affix}_train

steps/compute_cmvn_stats.sh \
  data/train_reseg${affix} exp/make_mfcc/train_reseg${affix} mfcc
utils/fix_data_dir.sh data/train_reseg${affix}

rm -r data/train_reseg${affix}/split20 || true
steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
  data/train_reseg${affix} data/lang_nosp exp/wsj_tri4a exp/wsj_tri4${affix}_ali_train_reseg${affix} || exit 1;

steps/train_sat.sh  --cmd "$train_cmd" 5000 100000 \
  data/train_reseg${affix} data/lang_nosp \
  exp/wsj_tri4${affix}_ali_train_reseg${affix} exp/tri4${affix} || exit 1;
  
utils/mkgraph.sh data/lang_nosp exp/tri4${affix}  exp/tri4${affix}/graph_nosp

for dset in dev test; do
  steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
    exp/tri4${affix}/graph_nosp data/${dset} exp/tri4${affix}/decode_${dset}
  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_nosp data/lang_nosp_rescore \
    data/${dset} exp/tri4${affix}/decode_${dset} \
    exp/tri4${affix}/decode_${dset}_rescore
done

new_affix=`echo $affix | perl -ne 'm/(\S+)([0-9])(\S+)/; print $1 . ($2+1) . $3;'`

###
# STAGE 1
###

steps/cleanup/segment_long_utterances.sh \
  --cmd "$train_cmd" --nj 80 \
  --stage $segment_stage \
  --max-bad-proportion 0.75 --align-full-hyp false \
  exp/tri4${affix} data/lang_nosp data/train_long data/train_reseg${new_affix} \
  exp/segment_long_utts${new_affix}_train

steps/compute_cmvn_stats.sh data/train_reseg${new_affix}
utils/fix_data_dir.sh data/train_reseg${new_affix}

steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
  data/train_reseg${new_affix} data/lang_nosp exp/tri4${affix} exp/tri4${affix}_ali_train_reseg${new_affix} || exit 1;

steps/train_sat.sh  --cmd "$train_cmd" 5000 100000 \
  data/train_reseg${new_affix} data/lang_nosp \
  exp/tri4${affix}_ali_train_reseg${new_affix} exp/tri5${new_affix} || exit 1;
  
utils/mkgraph.sh data/lang_nosp exp/tri5${new_affix} exp/tri5${new_affix}/graph_nosp

for dset in dev test; do
  steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
    exp/tri5${new_affix}/graph_nosp data/${dset} exp/tri5${new_affix}/decode_nosp_${dset}
  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_nosp data/lang_nosp_rescore \
    data/${dset} exp/tri5${new_affix}/decode_nosp_${dset} \
    exp/tri5${new_affix}/decode_nosp_${dset}_rescore
done

###
# STAGE 2
###

steps/get_prons.sh --cmd "$train_cmd" data/train_reseg${new_affix} \
  data/lang_nosp exp/tri5${new_affix}
utils/dict_dir_add_pronprobs.sh --max-normalize true \
  data/local/dict_nosp exp/tri5${new_affix}/pron_counts_nowb.txt \
  exp/tri5${new_affix}/sil_counts_nowb.txt \
  exp/tri5${new_affix}/pron_bigram_counts_nowb.txt data/local/dict

utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang
cp -rT data/lang data/lang_rescore
cp data/lang_nosp/G.fst data/lang/
cp data/lang_nosp_rescore/G.carpa data/lang_rescore/

utils/mkgraph.sh data/lang exp/tri5${new_affix} exp/tri5${new_affix}/graph

for dset in dev test; do
  steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
    exp/tri5${new_affix}/graph data/${dset} exp/tri5${new_affix}/decode_${dset}
  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
     data/${dset} exp/tri5${new_affix}/decode_${dset} \
     exp/tri5${new_affix}/decode_${dset}_rescore
done

###
# STAGE 3
###

srcdir=exp/tri5${new_affix}
cleanup_affix=cleaned
cleaned_data=data/train_reseg${new_affix}_${cleanup_affix}
cleaned_dir=${srcdir}_${cleanup_affix}

steps/cleanup/clean_and_segment_data.sh --stage $cleanup_stage --nj 80 \
  --cmd "$train_cmd" \
  data/train_reseg${new_affix} data/lang_nosp $srcdir \
  ${srcdir}_${cleanup_affix}_work $cleaned_data

steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
  $cleaned_data data/lang_nosp $srcdir ${srcdir}_ali_${cleanup_affix}

steps/train_sat.sh --cmd "$train_cmd" \
  5000 100000 $cleaned_data data/lang_nosp ${srcdir}_ali_${cleanup_affix} \
  ${cleaned_dir}

utils/mkgraph.sh data/lang_nosp $cleaned_dir ${cleaned_dir}/graph_nosp

for dset in dev test; do
  steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
    ${cleaned_dir}/graph_nosp data/${dset} ${cleaned_dir}/decode_nosp_${dset}
  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_nosp data/lang_nosp_rescore \
     data/${dset} ${cleaned_dir}/decode_nosp_${dset} \
     ${cleaned_dir}/decode_nosp_${dset}_rescore
done

exit 0

# Baseline | %WER 17.9 | 507 17783 | 85.1 10.5 4.4 3.0 17.9 90.9 | -0.055 | exp/tri3_cleaned/decode_dev_rescore/score_15_0.0/ctm.filt.filt.sys
# STAGE 0 | %WER 19.3 | 507 17783 | 83.9 10.8 5.2 3.2 19.3 92.3 | -0.178 | exp/tri4_1e/decode_dev_rescore/score_14_0.0/ctm.filt.filt.sys
# STAGE 1 | %WER 18.8 | 507 17783 | 84.4 10.7 4.9 3.2 18.8 91.7 | -0.199 | exp/tri5_2e/decode_nosp_dev_rescore/score_13_0.0/ctm.filt.filt.sys
# STAGE 2 | %WER 18.4 | 507 17783 | 84.7 10.4 4.8 3.2 18.4 91.7 | -0.192 | exp/tri5_2e/decode_dev_rescore/score_14_0.5/ctm.filt.filt.sys
# STAGE 3 | %WER 18.8 | 507 17783 | 84.4 10.7 4.9 3.2 18.8 91.3 | -0.162 | exp/tri5_2e_cleaned/decode_nosp_dev_rescore/score_14_0.0/ctm.filt.filt.sys

# Baseline | %WER 16.6 | 1155 27500 | 85.8 10.9 3.4 2.4 16.6 86.4 | -0.058 | exp/tri3_cleaned/decode_test_rescore/score_15_0.0/ctm.filt.filt.sys
# STAGE 0 | %WER 18.0 | 1155 27500 | 84.4 11.7 3.9 2.4 18.0 87.5 | -0.038 | exp/tri4_1e/decode_test_rescore/score_13_0.0/ctm.filt.filt.sys
# STAGE 1 | %WER 17.7 | 1155 27500 | 84.7 11.4 3.9 2.3 17.7 87.0 | -0.044 | exp/tri5_2e/decode_nosp_test_rescore/score_13_0.0/ctm.filt.filt.sys
# STAGE 2 | %WER 16.8 | 1155 27500 | 85.7 11.0 3.3 2.5 16.8 86.6 | -0.066 | exp/tri5_2e/decode_test_rescore/score_14_0.0/ctm.filt.filt.sys
# STAGE 3 | %WER 17.4 | 1155 27500 | 85.0 11.4 3.7 2.4 17.4 87.2 | -0.086 | exp/tri5_2e_cleaned/decode_nosp_test_rescore/score_12_0.0/ctm.filt.filt.sys

# Baseline | %WER 19.0 | 507 17783 | 83.9 11.4 4.7 2.9 19.0 92.1 | -0.054 | exp/tri3_cleaned/decode_dev/score_13_0.5/ctm.filt.filt.sys
# STAGE 0 | %WER 20.5 | 507 17783 | 82.8 11.9 5.3 3.3 20.5 94.1 | -0.098 | exp/tri4_1e/decode_dev/score_14_0.0/ctm.filt.filt.sys
# STAGE 1 | %WER 19.8 | 507 17783 | 83.3 11.5 5.2 3.2 19.8 94.7 | -0.133 | exp/tri5_2e/decode_nosp_dev/score_14_0.0/ctm.filt.filt.sys
# STAGE 2 | %WER 19.5 | 507 17783 | 83.9 11.4 4.7 3.5 19.5 94.1 | -0.120 | exp/tri5_2e/decode_dev/score_16_0.0/ctm.filt.filt.sys
# STAGE 3 | %WER 20.0 | 507 17783 | 83.5 11.7 4.8 3.5 20.0 93.3 | -0.111 | exp/tri5_2e_cleaned/decode_nosp_dev/score_13_0.0/ctm.filt.filt.sys

# Baseline | %WER 17.6 | 1155 27500 | 84.8 11.7 3.5 2.4 17.6 87.6 | 0.001 | exp/tri3_cleaned/decode_test/score_15_0.0/ctm.filt.filt.sys
# STAGE 0 | %WER 19.1 | 1155 27500 | 83.4 12.5 4.1 2.5 19.1 88.6 | 0.022 | exp/tri4_1e/decode_test/score_13_0.0/ctm.filt.filt.sys
# STAGE 1 | %WER 18.7 | 1155 27500 | 83.7 12.2 4.1 2.4 18.7 88.1 | 0.007 | exp/tri5_2e/decode_nosp_test/score_13_0.0/ctm.filt.filt.sys
# STAGE 2 | %WER 17.9 | 1155 27500 | 84.8 11.9 3.3 2.7 17.9 87.5 | -0.015 | exp/tri5_2e/decode_test/score_13_0.0/ctm.filt.filt.sys
# STAGE 3 | %WER 18.4 | 1155 27500 | 83.9 12.1 4.0 2.3 18.4 88.1 | -0.015 | exp/tri5_2e_cleaned/decode_nosp_test/score_13_0.0/ctm.filt.filt.sys
