#! /bin/bash

. ./cmd.sh
. ./path.sh

set -e -o pipefail -u

segment_stage=-4
affix=_1c

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
  exp/wsj_tri4a data/lang_nosp data/train_long data/train_long/text data/train_reseg${affix} \
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

decode_nj=30
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
  exp/tri4${affix} data/lang_nosp data/train_long data/train_long/text data/train_reseg${new_affix} \
  exp/segment_long_utts${new_affix}_train

steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
  data/train_reseg${new_affix} data/lang_nosp exp/tri4${affix} exp/tri4${affix}_ali_train_reseg${new_affix} || exit 1;

steps/train_sat.sh  --cmd "$train_cmd" 5000 100000 \
  data/train_reseg${new_affix} data/lang_nosp \
  exp/tri4a_ali_train_reseg${new_affix} exp/tri5a${new_affix} || exit 1;
  
utils/mkgraph.sh data/lang_nosp exp/tri5a${new_affix} exp/tri5a${new_affix}/graph_nosp

exit 0
