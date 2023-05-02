#!/usr/bin/env bash

# Copyright 2019  IBM (Michael Picheny) Adapted from AMI recipe for MALACH Corpus
# Copyright 2016  Vimal Manohar
#           2016  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

# This script demonstrates how to re-segment training data selecting only the
# "good" audio that matches the transcripts.
# The basic idea is to decode with an existing in-domain acoustic model, and a
# biased language model built from the reference, and then work out the
# segmentation from a ctm like file.


set -e
set -o pipefail
set -u

stage=0
cleanup_stage=0
train_stage=-10
cleanup_affix=cleaned
nj=50
gmm=tri3
lang=data/lang


. ./path.sh
. ./cmd.sh
. utils/parse_options.sh

data=data/train
cleaned_data=data/train_${cleanup_affix}
srcdir=exp/$gmm
dir=exp/${gmm}_${cleanup_affix}_work

if [ $stage -le 1 ]; then
  # This does the actual data cleanup.
  # Note: using the shorter min-length options "--min-segment-length 0.3 --min-new-segment-length 0.6" \
  # [vs. default 0.5, 0.1] leads to more data being kept, I think about 92% vs 88%; and
  # the WER changes were inconsistent but overall very slightly better, e.g. (0.2% better, 0.1% worse)
  # on different test sets.
  steps/cleanup/clean_and_segment_data.sh --stage $cleanup_stage --nj $nj --cmd "$train_cmd" \
      --segmentation-opts "--min-segment-length 0.3 --min-new-segment-length 0.6" \
    $data $lang $srcdir $dir $cleaned_data
fi

if [ $stage -le 3 ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/train_${cleanup_affix} data/lang exp/$gmm exp/${gmm}_ali_${cleanup_affix}
fi

if [ $stage -le 4 ]; then
  steps/train_sat.sh --cmd "$train_cmd" --stage "$train_stage" \
    5000 80000 data/train_${cleanup_affix} data/lang exp/${gmm}_ali_${cleanup_affix} exp/${gmm}_${cleanup_affix}
fi

[ ! -r data/local/lm/final_lm ] && echo "Please, run 'run_prepare_shared.sh' first!" && exit 1
final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-9


if [ $stage -le 5 ]; then
  graph_dir=exp/${gmm}_${cleanup_affix}/graph_$LM
  nj_dev=$(cat data/dev/spk2utt | wc -l)

  $decode_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_$LM exp/${gmm}_${cleanup_affix} $graph_dir
  steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
    $graph_dir data/dev exp/${gmm}_${cleanup_affix}/decode_dev_$LM
fi

