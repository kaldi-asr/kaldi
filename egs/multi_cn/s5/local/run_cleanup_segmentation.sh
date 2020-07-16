#!/usr/bin/env bash

# Copyright 2016  Vimal Manohar
#           2016  Yiming Wang
#           2016  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

# This script demonstrates how to re-segment training data selecting only the
# "good" audio that matches the transcripts.
# The basic idea is to decode with an existing in-domain acoustic model, and a
# biased language model built from the reference, and then work out the
# segmentation from a ctm like file.

# For nnet3 and chain results after cleanup, see the scripts in
# local/nnet3/run_tdnn.sh and local/chain/run_tdnn_6z.sh

# GMM Results for speaker-independent (SI) and speaker adaptive training (SAT) systems on dev and test sets
# [will add these later].

set -e
set -o pipefail
set -u

stage=0
cleanup_stage=0
data=data/train_all
cleanup_affix=cleaned
srcdir=exp/tri4a
nj=100
decode_nj=10
decode_num_threads=4
test_sets=""
corpus_lm=false

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

cleaned_data=${data}_${cleanup_affix}

dir=${srcdir}_${cleanup_affix}_work
cleaned_dir=${srcdir}_${cleanup_affix}

if [ $stage -le 1 ]; then
  # This does the actual data cleanup.
  steps/cleanup/clean_and_segment_data.sh --stage $cleanup_stage --nj $nj --cmd "$train_cmd" \
    $data data/lang $srcdir $dir $cleaned_data
fi

if [ $stage -le 2 ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    $cleaned_data data/lang $srcdir ${srcdir}_ali_${cleanup_affix}
fi

if [ $stage -le 3 ]; then
  steps/train_sat.sh --cmd "$train_cmd" \
    12000 190000 $cleaned_data data/lang ${srcdir}_ali_${cleanup_affix} ${cleaned_dir}
fi

if [ $stage -le 4 ]; then
  # Test with the models trained on cleaned-up data.
  utils/mkgraph.sh data/lang_combined_tg ${cleaned_dir} ${cleaned_dir}/graph_tg

  for c in $test_sets; do
    (
      steps/decode_fmllr.sh --nj $decode_nj --num-threads $decode_num_threads \
        --cmd "$decode_cmd" \
        ${cleaned_dir}/graph_tg data/${c}/test ${cleaned_dir}/decode_${c}_tg
      if $corpus_lm; then
        $mkgraph_cmd ${cleaned_dir}/log/mkgraph.$c.log \
          utils/mkgraph.sh data/lang_${c}_tg ${cleaned_dir} ${cleaned_dir}/graph_$c || exit 1;
        steps/decode_fmllr.sh --nj $decode_nj --num-threads $decode_num_threads \
          --cmd "$decode_cmd" \
          ${cleaned_dir}/graph_$c data/$c/test ${cleaned_dir}/decode_${c}_test_clm || exit 1;
      fi
   ) &
  done
fi

wait;
exit 0;
