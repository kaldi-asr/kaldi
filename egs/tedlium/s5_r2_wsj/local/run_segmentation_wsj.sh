#! /bin/bash

. ./cmd.sh
. ./path.sh

set -e -o pipefail -u

segment_stage=7
affix=_2b

###############################################################################
# Segment long recordings using TF-IDF retrieval of reference text 
# for uniformly segmented audio chunks based on Smith-Waterman alignment.
# Use a model trained on train_si84 (tri2b)
###############################################################################

false && {
utils/data/convert_data_dir_to_whole.sh data/train data/train_long
steps/make_mfcc.sh --nj 40 --cmd "$train_cmd" \
  data/train_long exp/make_mfcc/train_long mfcc
steps/compute_cmvn_stats.sh \
  data/train_long exp/make_mfcc/train_long mfcc
utils/fix_data_dir.sh data/train_long

bash -x steps/cleanup/segment_long_utterances.sh \
  --cmd "$train_cmd" \
  --stage $segment_stage \
  --config conf/segment_long_utts.conf \
  --max-segment-duration 30 --overlap-duration 5 \
  --num-neighbors-to-search 1 --nj 80 \
  exp/wsj_tri2b data/lang_nosp data/train_long data/train_long/text data/train_reseg${affix} \
  exp/segment_wsj_long_utts${affix}_train

steps/compute_cmvn_stats.sh \
  data/train_reseg${affix} exp/make_mfcc/train_reseg${affix} mfcc
utils/fix_data_dir.sh data/train_reseg${affix}

rm -rf data/train_reseg${affix}/split20
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
}

new_affix=`echo $affix | perl -ne 'm/(\S+)([0-9])(\S+)/; print $1 . ($2+1) . $3;'`

false && {
bash -x steps/cleanup/segment_long_utterances.sh \
  --cmd "$train_cmd" \
  --stage $segment_stage \
  --config conf/segment_long_utts.conf \
  --max-segment-duration 30 --overlap-duration 5 \
  --num-neighbors-to-search 1 --nj 80 \
  exp/tri4${affix} data/lang_nosp data/train_long data/train_long/text data/train_reseg${new_affix} \
  exp/segment_long_utts${new_affix}_train

steps/compute_cmvn_stats.sh data/train_reseg${new_affix}
utils/fix_data_dir.sh data/train_reseg${new_affix}

steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
  data/train_reseg${new_affix} data/lang_nosp exp/tri4${affix} exp/tri4${affix}_ali_train_reseg${new_affix} || exit 1;

steps/train_sat.sh  --cmd "$train_cmd" 5000 100000 \
  data/train_reseg${new_affix} data/lang_nosp \
  exp/tri4${affix}_ali_train_reseg${new_affix} exp/tri5${new_affix} || exit 1;
}
  
utils/mkgraph.sh data/lang_nosp exp/tri5${new_affix} exp/tri5${new_affix}/graph_nosp

decode_nj=30
for dset in dev test; do
  steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
    exp/tri5${new_affix}/graph_nosp data/${dset} exp/tri5${new_affix}/decode_nosp_${dset}
  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_nosp data/lang_nosp_rescore \
    data/${dset} exp/tri5${new_affix}/decode_nosp_${dset} \
    exp/tri5${new_affix}/decode_nosp_${dset}_rescore
done

exit 0
