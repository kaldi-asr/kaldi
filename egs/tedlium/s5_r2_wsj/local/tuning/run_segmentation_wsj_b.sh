#! /bin/bash

. ./cmd.sh
. ./path.sh

set -e -o pipefail -u

segment_stage=7
affix=_2b
decode_nj=30
cleanup_stage=8

###############################################################################
# Segment long recordings using TF-IDF retrieval of reference text 
# for uniformly segmented audio chunks based on Smith-Waterman alignment.
# Use a model trained on train_si84 (tri2b)
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
  exp/wsj_tri2b data/lang_nosp data/train_long data/train_long/text data/train_reseg${affix} \
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
  --max-bad-proportion 0.75 --align-full-hyp true \
  exp/tri4${affix} data/lang_nosp data/train_long data/train_long/text data/train_reseg${new_affix} \
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
# STAGE 0 | %WER 19.4 | 507 17783 | 84.0 11.0 4.9 3.4 19.4 92.3 | -0.182 | exp/tri4_2b/decode_dev_rescore/score_13_0.0/ctm.filt.filt.sys
# STAGE 1 | %WER 18.8 | 507 17783 | 84.5 10.4 5.0 3.4 18.8 91.5 | -0.216 | exp/tri5_3b/decode_nosp_dev_rescore/score_13_0.0/ctm.filt.filt.sys
# STAGE 2 | %WER 18.3 | 507 17783 | 85.2 10.5 4.3 3.5 18.3 90.3 | -0.184 | exp/tri5_3b/decode_dev_rescore/score_14_0.0/ctm.filt.filt.sys
# STAGE 3 | %WER 18.8 | 507 17783 | 84.3 10.6 5.1 3.1 18.8 91.5 | -0.186 | exp/tri5_3b_cleaned/decode_nosp_dev_rescore/score_15_0.0/ctm.filt.filt.sys

# Baseline | %WER 16.6 | 1155 27500 | 85.8 10.9 3.4 2.4 16.6 86.4 | -0.058 | exp/tri3_cleaned/decode_test_rescore/score_15_0.0/ctm.filt.filt.sys
# STAGE 0 | %WER 18.1 | 1155 27500 | 84.3 11.8 4.0 2.4 18.1 87.3 | -0.036 | exp/tri4_2b/decode_test_rescore/score_13_0.0/ctm.filt.filt.sys
# STAGE 1 | %WER 17.7 | 1155 27500 | 84.6 11.5 3.9 2.3 17.7 87.3 | -0.053 | exp/tri5_3b/decode_nosp_test_rescore/score_13_0.0/ctm.filt.filt.sys
# STAGE 2 | %WER 17.0 | 1155 27500 | 85.7 11.2 3.2 2.6 17.0 86.7 | -0.075 | exp/tri5_3b/decode_test_rescore/score_13_0.0/ctm.filt.filt.sys
# STAGE 3 | %WER 17.7 | 1155 27500 | 84.7 11.6 3.7 2.4 17.7 87.9 | -0.082 | exp/tri5_3b_cleaned/decode_nosp_test_rescore/score_12_0.0/ctm.filt.filt.sys

# Baseline | %WER 19.0 | 507 17783 | 83.9 11.4 4.7 2.9 19.0 92.1 | -0.054 | exp/tri3_cleaned/decode_dev/score_13_0.5/ctm.filt.filt.sys
# STAGE 0 | %WER 20.5 | 507 17783 | 82.9 11.8 5.3 3.4 20.5 93.5 | -0.120 | exp/tri4_2b/decode_dev/score_14_0.0/ctm.filt.filt.sys
# STAGE 1 | %WER 20.1 | 507 17783 | 83.1 11.6 5.3 3.3 20.1 93.5 | -0.116 | exp/tri5_3b/decode_nosp_dev/score_14_0.0/ctm.filt.filt.sys
# STAGE 2 | %WER 19.5 | 507 17783 | 84.0 11.4 4.6 3.5 19.5 92.9 | -0.115 | exp/tri5_3b/decode_dev/score_15_0.0/ctm.filt.filt.sys
# STAGE 3 | %WER 20.0 | 507 17783 | 83.6 11.6 4.8 3.6 20.0 93.7 | -0.128 | exp/tri5_3b_cleaned/decode_nosp_dev/score_13_0.0/ctm.filt.filt.sys

# Baseline | %WER 17.6 | 1155 27500 | 84.8 11.7 3.5 2.4 17.6 87.6 | 0.001 | exp/tri3_cleaned/decode_test/score_15_0.0/ctm.filt.filt.sys
# STAGE 0 | %WER 19.0 | 1155 27500 | 83.3 12.3 4.4 2.3 19.0 88.6 | 0.016 | exp/tri4_2b/decode_test/score_14_0.0/ctm.filt.filt.sys
# STAGE 1 | %WER 18.8 | 1155 27500 | 83.6 12.3 4.1 2.4 18.8 88.0 | -0.001 | exp/tri5_3b/decode_nosp_test/score_13_0.0/ctm.filt.filt.sys
# STAGE 2 | %WER 18.1 | 1155 27500 | 84.5 12.1 3.4 2.6 18.1 88.2 | -0.005 | exp/tri5_3b/decode_test/score_14_0.0/ctm.filt.filt.sys
# STAGE 3 | %WER 18.6 | 1155 27500 | 83.6 12.2 4.2 2.2 18.6 88.5 | -0.010 | exp/tri5_3b_cleaned/decode_nosp_test/score_14_0.0/ctm.filt.filt.sys
