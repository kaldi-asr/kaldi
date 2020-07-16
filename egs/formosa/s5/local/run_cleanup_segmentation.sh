#!/usr/bin/env bash

# Copyright   2016  Vimal Manohar
#             2016  Johns Hopkins University (author: Daniel Povey)
#             2017  Nagendra Kumar Goel
#             2019  AsusTek Computer Inc. (author: Alex Hung)
# Apache 2.0

# This script demonstrates how to re-segment training data selecting only the
# "good" audio that matches the transcripts.
# The basic idea is to decode with an existing in-domain acoustic model, and a
# biased language model built from the reference, and then work out the
# segmentation from a ctm like file.

# For nnet3 and chain results after cleanup, see the scripts in
# local/nnet3/run_tdnn.sh and local/chain/run_tdnn.sh

# GMM Results for speaker-independent (SI) and speaker adaptive training (SAT) systems on dev and test sets
# [will add these later].

set -e
set -o pipefail
set -u

stage=0
cleanup_stage=0
data=data/train
cleanup_affix=cleaned
srcdir=exp/tri5a
langdir=data/lang_test
nj=20
decode_nj=20
decode_num_threads=1

. ./cmd.sh
if [ -f ./path.sh ]; then . ./path.sh; fi
. utils/parse_options.sh

cleaned_data=${data}_${cleanup_affix}

dir=${srcdir}_${cleanup_affix}_work
cleaned_dir=${srcdir}_${cleanup_affix}

if [ $stage -le 1 ]; then
  # This does the actual data cleanup.
  steps/cleanup/clean_and_segment_data.sh --stage $cleanup_stage \
    --nj $nj --cmd "$train_cmd" \
    $data $langdir $srcdir $dir $cleaned_data
fi

if [ $stage -le 2 ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    $cleaned_data $langdir $srcdir ${srcdir}_ali_${cleanup_affix}
fi

if [ $stage -le 3 ]; then
  steps/train_sat.sh --cmd "$train_cmd" \
    3500 100000 $cleaned_data $langdir ${srcdir}_ali_${cleanup_affix} ${cleaned_dir}
fi

utils/data/get_utt2dur.sh data/train_cleaned
ori_avg_dur=$(awk 'BEGIN{total=0}{total += $2}END{printf("%.2f", total/NR)}' ${data}/utt2dur)
new_avg_dur=$(awk 'BEGIN{total=0}{total += $2}END{printf("%.2f", total/NR)}' ${cleaned_data}/utt2dur)
echo "average duration was reduced from ${ori_avg_dur}s to ${new_avg_dur}s."
# average duration was reduced from 21.68s to 10.97s.
exit 0;
