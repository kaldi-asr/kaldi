#!/bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0

set -e
set -o pipefail

# This script demonstrates how to re-segment long audios into short segments.
# The basic idea is to decode with an existing in-domain acoustic model, and a
# bigram language model built from the reference, and then work out the
# segmentation from a ctm like file.
# This is similart to _a, but uses a automatically segmented data directory.

. ./cmd.sh
. ./path.sh

segment_stage=-8
nj=40
reco_nj=80
affix=e
new_affix=2e

. utils/parse_options.sh

false && {
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
}

###############################################################################
# Segment long recordings using TF-IDF retrieval of reference text 
# for uniformly segmented audio chunks based on Smith-Waterman alignment.
# Use a SAT model trained on train_si284 (wsj_tri3)
###############################################################################

true && {
bash -x steps/cleanup/segment_long_utterances.sh --cmd "$train_cmd" \
  --stage $segment_stage \
  --config conf/segment_long_utts.conf --align-full-hyp false \
  --max-segment-duration 30 --overlap-duration 5 \
  --num-neighbors-to-search 1 --nj $reco_nj \
  exp/wsj_tri3 data/lang_nosp data/train_long.seg_lstm_1e_sad_music data/train_long/text \
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
}

###############################################################################
# Segment long recordings using TF-IDF retrieval of reference text 
# for uniformly segmented audio chunks based on Smith-Waterman alignment.
# Use a SAT model trained on tri4_a
###############################################################################

true && {
steps/cleanup/segment_long_utterances.sh --cmd "$train_cmd" \
  --stage $segment_stage \
  --config conf/segment_long_utts.conf --align-full-hyp false \
  --max-segment-duration 30 --overlap-duration 5 \
  --num-neighbors-to-search 1 --nj $reco_nj \
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
}

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

###############################################################################
# Train new model on segmented data directory starting from the same model
# used for segmentation. (tri2b)
###############################################################################

# Align tri2b system with reseg${affix} data
steps/align_si.sh  --nj 40 --cmd "$train_cmd" \
  data/train_reseg_${affix} \
  data/lang_nosp exp/wsj_tri2b exp/wsj_tri2b_ali_reseg_${affix}  || exit 1;

steps/train_deltas.sh --cmd "$train_cmd" 2000 10000 \
  data/train_reseg_${affix} data/lang_nosp exp/tri1${affix}

steps/align_si.sh --nj 40 --cmd "$train_cmd" \
  data/train_reseg_${affix} \
  data/lang_nosp exp/tri1${affix} exp/tri1${affix}_ali_reseg_${affix}

steps/train_lda_mllt.sh --cmd "$train_cmd" 3500 25000 \
  data/train_reseg_${affix} data/lang_nosp exp/tri2${affix}

affix=d
steps/cleanup/segment_long_utterances.sh --cmd "$train_cmd" \
  --stage $segment_stage \
  --config conf/segment_long_utts.conf \
  --max-segment-duration 30 --overlap-duration 5 \
  --num-neighbors-to-search 1 --nj 80 \
  exp/tri2a data/lang_nosp data/train_long data/train_reseg_${affix} \
  exp/segment_long_utts_${affix}_train

steps/compute_cmvn_stats.sh data/train_reseg_${affix} \
  exp/make_mfcc/train_reseg_${affix} mfcc
utils/fix_data_dir.sh data/train_reseg_${affix}

###############################################################################
# Train new model on segmented data directory starting from the same model
# used for segmentation. (tri2b)
###############################################################################

steps/align_si.sh --nj 40 --cmd "$train_cmd" \
  data/train_reseg_${affix} \
  data/lang_nosp exp/tri2b exp/tri2b_ali_reseg_${affix}  || exit 1;

# Train SAT system on reseg data
steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
  data/train_reseg_${affix} data/lang_nosp \
  exp/tri2b_ali_reseg_${affix} exp/tri3${affix}

(
utils/mkgraph.sh data/lang_nosp_test_tgpr \
  exp/tri3${affix} exp/tri3${affix}/graph_nosp || exit 1;
for dset in eval98.pem eval97.pem eval99_1.pem eval99_2.pem; do
  this_nj=`cat data/$dset/spk2utt | wc -l`
  if [ $this_nj -gt 20 ]; then
    this_nj=20
  fi
  steps/decode_fmllr.sh --nj $this_nj --cmd "$decode_cmd" --num-threads 4 \
    exp/tri3${affix}/graph_nosp data/$dset exp/tri3${affix}/decode_nosp_${dset}
  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
    data/lang_nosp_test data/lang_nosp_test_rescore \
    data/${dset} exp/tri3${affix}/decode_nosp_${dset} \
    exp/tri3${affix}/decode_nosp_${dset}_rescore
done
) &

exit 0

steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
  data/train data/lang_nosp exp/tri3${affix} exp/tri3${affix}_ali

steps/train_sat.sh --cmd "$train_cmd" 5000 100000 \
  data/train data/lang_nosp exp/tri3${affix}_ali exp/tri4${affix}

(
utils/mkgraph.sh data/lang_nosp_test exp/tri4${affix} \
  exp/tri4${affix}/graph_nosp
for dset in eval98.pem eval97.pem eval99_1.pem eval99_2.pem; do
  this_nj=`cat data/$dset/spk2utt | wc -l`
  if [ $this_nj -gt 20 ]; then
    this_nj=20
  fi
  steps/decode_fmllr.sh --nj $this_nj --cmd "$decode_cmd" --num-threads 4 \
    exp/tri4${affix}/graph_nosp data/$dset exp/tri4${affix}/decode_nosp_${dset}
  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
    data/lang_nosp_test data/lang_nosp_test_rescore \
    data/${dset} exp/tri4${affix}/decode_nosp_${dset} \
    exp/tri4${affix}/decode_nosp_${dset}_rescore
done
) &

exit 0

###############################################################################
# Train new model on segmented data directory starting from a better model
# (tri3b)
###############################################################################

# Align tri3b system with reseg data
steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
  data/train_si284_reseg${affix} data/lang_nosp exp/tri3b \
  exp/tri3b_ali_si284_reseg${affix}

# Train SAT system on reseg data
steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
  data/train_si284_reseg${affix} data/lang_nosp \
  exp/tri3b_ali_si284_reseg${affix} exp/tri4c_reseg${affix}

(
 utils/mkgraph.sh data/lang_nosp_test_tgpr \
   exp/tri4c_reseg${affix} exp/tri4c_reseg${affix}/graph_nosp_tgpr || exit 1;
 steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
   exp/tri4c_reseg${affix}/graph_nosp_tgpr data/test_dev93 \
   exp/tri4c_reseg${affix}/decode_nosp_tgpr_dev93 || exit 1;
 steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
   exp/tri4c_reseg${affix}/graph_nosp_tgpr data/test_eval92 \
   exp/tri4c_reseg${affix}/decode_nosp_tgpr_eval92 || exit 1;
) &

###############################################################################
# cleaned_a : Cleanup the segmented data directory using tri3b model.
###############################################################################

steps/cleanup/clean_and_segment_data.sh --cmd "$train_cmd" \
  --nj 80 \
  data/train_si284_reseg${affix} data/lang_nosp \
  exp/tri3b_ali_si284_reseg${affix} exp/tri3b_work_si284_reseg${affix} data/train_si284_reseg${affix}_cleaned_a

###############################################################################
# Train new model on the cleaned_a data directory
###############################################################################

# Align tri3b system with cleaned data
steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
  data/train_si284_reseg${affix}_cleaned_a data/lang_nosp exp/tri3b \
  exp/tri3b_ali_si284_reseg${affix}_cleaned_a

# Train SAT system on cleaned data
steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
  data/train_si284_reseg${affix}_cleaned_a data/lang_nosp \
  exp/tri3b_ali_si284_reseg${affix}_cleaned_a exp/tri4d${affix}_cleaned_a

(
 utils/mkgraph.sh data/lang_nosp_test_tgpr \
   exp/tri4d${affix}_cleaned_a exp/tri4d${affix}_cleaned_a/graph_nosp_tgpr || exit 1;
 steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
   exp/tri4d${affix}_cleaned_a/graph_nosp_tgpr data/test_dev93 \
   exp/tri4d${affix}_cleaned_a/decode_nosp_tgpr_dev93 || exit 1;
 steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
   exp/tri4d${affix}_cleaned_a/graph_nosp_tgpr data/test_eval92 \
   exp/tri4d${affix}_cleaned_a/decode_nosp_tgpr_eval92 || exit 1;
) &

###############################################################################
# cleaned_b : Cleanup the segmented data directory using the tri3c_reseg
# model, which is a like a first-pass model trained on the resegmented data.
###############################################################################

steps/cleanup/clean_and_segment_data.sh --cmd "$train_cmd" \
  --nj 80 \
  data/train_si284_reseg${affix} data/lang_nosp \
  exp/tri3c_reseg${affix} exp/tri3c_reseg${affix}_work_si284_reseg${affix} \
  data/train_si284_reseg${affix}_cleaned_b

###############################################################################
# Train new model on the cleaned_b data directory
###############################################################################

# Align tri3c_reseg system with cleaned data
steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
  data/train_si284_reseg${affix}_cleaned_b data/lang_nosp exp/tri3c_reseg${affix} \
  exp/tri3c_reseg${affix}_ali_si284_reseg${affix}_cleaned_b

# Train SAT system on cleaned data
steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
  data/train_si284_reseg${affix}_cleaned_b data/lang_nosp \
  exp/tri3c_reseg${affix}_ali_si284_reseg${affix}_cleaned_b exp/tri4d${affix}_cleaned_b

(
 utils/mkgraph.sh data/lang_nosp_test_tgpr \
   exp/tri4d${affix}_cleaned_b exp/tri4d${affix}_cleaned_b/graph_nosp_tgpr || exit 1;
 steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
   exp/tri4d${affix}_cleaned_b/graph_nosp_tgpr data/test_dev93 \
   exp/tri4d${affix}_cleaned_b/decode_nosp_tgpr_dev93 || exit 1;
 steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
   exp/tri4d${affix}_cleaned_b/graph_nosp_tgpr data/test_eval92 \
   exp/tri4d${affix}_cleaned_b/decode_nosp_tgpr_eval92 || exit 1;
) &

###############################################################################
# cleaned_c : Cleanup the segmented data directory using the tri4c_reseg
# model, which is a like a first-pass model trained on the resegmented data.
###############################################################################

steps/cleanup/clean_and_segment_data.sh --cmd "$train_cmd" \
  --nj 80 \
  data/train_si284_reseg${affix} data/lang_nosp \
  exp/tri4c_reseg${affix} exp/tri4c_reseg${affix}_work_si284_reseg${affix} \
  data/train_si284_reseg${affix}_cleaned_c

###############################################################################
# Train new model on the cleaned_c data directory
###############################################################################

# Align tri4c_reseg system with cleaned data
steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
  data/train_si284_reseg${affix}_cleaned_c data/lang_nosp exp/tri4c_reseg${affix} \
  exp/tri4c_reseg${affix}_ali_si284_reseg${affix}_cleaned_c

# Train SAT system on cleaned data
steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
  data/train_si284_reseg${affix}_cleaned_c data/lang_nosp \
  exp/tri4c_reseg${affix}_ali_si284_reseg${affix}_cleaned_c exp/tri4d${affix}_cleaned_c

(
 utils/mkgraph.sh data/lang_nosp_test_tgpr \
   exp/tri4d${affix}_cleaned_c exp/tri4d${affix}_cleaned_c/graph_nosp_tgpr || exit 1;
 steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
   exp/tri4d${affix}_cleaned_c/graph_nosp_tgpr data/test_dev93 \
   exp/tri4d${affix}_cleaned_c/decode_nosp_tgpr_dev93 || exit 1;
 steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
   exp/tri4d${affix}_cleaned_c/graph_nosp_tgpr data/test_eval92 \
   exp/tri4d${affix}_cleaned_c/decode_nosp_tgpr_eval92 || exit 1;
) &

