#!/usr/bin/env bash

# Copyright 2016  Vimal Manohar
#           2016  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

# This script demonstrates how to re-segment training data selecting only the
# "good" audio that matches the transcripts.
# The basic idea is to decode with an existing in-domain acoustic model, and a
# biased language model built from the reference, and then work out the
# segmentation from a ctm like file.

stage=0
cleanup_stage=0
data=data/train
cleanup_affix=cleaned
srcdir=exp/tri3
nj=100
decode_nj=16
decode_num_threads=4

. ./path.sh
. ./cmd.sh

set -e
set -o pipefail
set -u

. utils/parse_options.sh

cleaned_data=${data}_${cleanup_affix}

dir=${srcdir}_${cleanup_affix}_work
cleaned_dir=${srcdir}_${cleanup_affix}

if [ $stage -le 1 ]; then
  # This does the actual data cleanup.
  steps/cleanup/clean_and_segment_data.sh --stage $cleanup_stage --nj $nj --cmd "$train_cmd" \
    $data data/lang_nosp $srcdir $dir $cleaned_data
fi

if [ $stage -le 2 ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    $cleaned_data data/lang_nosp $srcdir ${srcdir}_ali_${cleanup_affix}
fi

if [ $stage -le 3 ]; then
  steps/train_sat.sh --cmd "$train_cmd" \
    4200 40000 $cleaned_data data/lang_nosp ${srcdir}_ali_${cleanup_affix} ${cleaned_dir}
fi

if [ $stage -le 4 ]; then
  # Test with the model trained on cleaned-up data.
  utils/mkgraph.sh data/lang_nosp_test ${cleaned_dir} ${cleaned_dir}/graph_nosp

  for dset in eval97.pem eval98.pem eval99_1.pem eval99_2.pem; do
    this_nj=`cat data/$dset/spk2utt | wc -l`
    if [ $this_nj -gt $decode_nj ]; then
      this_nj=$decode_nj
    fi
    steps/decode_fmllr.sh --nj $decode_nj --num-threads $decode_num_threads \
       --cmd "$decode_cmd" \
       ${cleaned_dir}/graph_nosp data/${dset} ${cleaned_dir}/decode_nosp_${dset}
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_nosp_test data/lang_nosp_test_rescore \
       data/${dset} ${cleaned_dir}/decode_nosp_${dset} ${cleaned_dir}/decode_nosp_${dset}_rescore
  done
fi

if [ $stage -le 5 ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    $cleaned_data data/lang_nosp ${cleaned_dir} ${cleaned_dir}_ali_${cleanup_affix}
fi

if [ $stage -le 6 ]; then
  steps/train_sat.sh --cmd "$train_cmd" \
    5000 100000 $cleaned_data data/lang_nosp \
    ${cleaned_dir}_ali_${cleanup_affix} exp/tri4_${cleanup_affix}
fi

cleaned_dir=exp/tri4_${cleanup_affix}
if [ $stage -le 7 ]; then
  # Test with the larger model trained on cleaned-up data.
  utils/mkgraph.sh data/lang_nosp_test ${cleaned_dir} ${cleaned_dir}/graph_nosp

  for dset in eval97.pem eval98.pem eval99_1.pem eval99_2.pem; do
    this_nj=`cat data/$dset/spk2utt | wc -l`
    if [ $this_nj -gt $decode_nj ]; then
      this_nj=$decode_nj
    fi
    steps/decode_fmllr.sh --nj $decode_nj --num-threads $decode_num_threads \
       --cmd "$decode_cmd"  \
       ${cleaned_dir}/graph_nosp data/${dset} ${cleaned_dir}/decode_nosp_${dset}
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_nosp_test data/lang_nosp_test_rescore \
       data/${dset} ${cleaned_dir}/decode_nosp_${dset} ${cleaned_dir}/decode_nosp_${dset}_rescore
  done
fi
