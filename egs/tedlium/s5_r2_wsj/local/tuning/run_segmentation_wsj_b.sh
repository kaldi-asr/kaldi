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

# Config saved for reproduction of results
# # TF-IDF similarity search options
# max_words=1000
# num_neighbors_to_search=1
# neighbor_tfidf_threshold=0.5
#
# align_full_hyp=false
# 
# # first-pass segmentation opts
# min_segment_length=0.5
# min_new_segment_length=1.0
# max_tainted_length=0.05
# max_edge_silence_length=0.5
# max_edge_non_scored_length=0.5
# max_internal_silence_length=2.0
# max_internal_non_scored_length=2.0
# unk_padding=0.05
# max_junk_proportion=0.1
# min_split_point_duration=0.1
# max_deleted_words_kept_when_merging=1
# silence_factor=1
# incorrect_words_factor=1
# tainted_words_factor=1
# max_wer=50
# max_segment_length_for_merging=60
# max_bad_proportion=0.5
# max_intersegment_incorrect_words_length=1
# max_segment_length_for_splitting=10
# hard_max_segment_length=15
# min_silence_length_to_split_at=0.3
# min_non_scored_length_to_split_at=0.3

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
