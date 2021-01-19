#!/bin/bash
# Copyright   2019   David Snyder
#             2020   Desh Raj

# Apache 2.0.
#
# This script takes an input directory that has a segments file (and
# a feats.scp file), and performs diarization on it using spectral
# clustering.

stage=0
nj=10
cmd="run.pl"
rttm_affix=

echo "$0 $@"  # Print the command line for logging
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
if [ $# != 3 ]; then
  echo "Usage: $0 <model-dir> <in-data-dir> <out-dir>"
  echo "e.g.: $0 exp/xvector_nnet_1a  data/dev exp/dev_diarization"
  echo "Options: "
  echo "  --nj <nj>                                        # number of parallel jobs."
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  exit 1;
fi

model_dir=$1
data_in=$2
out_dir=$3

name=`basename $data_in`

for f in $data_in/feats.scp $data_in/segments \
  $model_dir/final.raw $model_dir/extract.config; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

if [ $stage -le 1 ]; then
  echo "$0: computing features for x-vector extractor"
  utils/fix_data_dir.sh data/${name}
  rm -rf data/${name}_cmn
  local/nnet3/xvector/prepare_feats.sh --nj $nj --cmd "$cmd" \
    data/$name data/${name}_cmn exp/${name}_cmn
  cp data/$name/segments exp/${name}_cmn/
  utils/fix_data_dir.sh data/${name}_cmn
fi

if [ $stage -le 2 ]; then
  echo "$0: extracting x-vectors for all segments"
  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$cmd" \
    --nj $nj --window 1.5 --period 0.75 --apply-cmn false \
    --min-segment 0.5 $model_dir \
    data/${name}_cmn $out_dir/xvectors_${name}
fi

# Perform cosine similarity scoring
if [ $stage -le 3 ]; then
  # Perform cosine similarity scoring on all pairs of segments for each recording.
  echo "$0: performing cosine similarity scoring between all pairs of x-vectors"
  diarization/score_cossim.sh --cmd "$cmd" \
    --nj $nj $out_dir/xvectors_${name} \
    $out_dir/xvectors_${name}/cossim_scores
fi

if [ $stage -le 4 ]; then
  echo "$0: performing spectral clustering using cosine similarity scores"
  diarization/scluster.sh --cmd "$cmd" --nj $nj \
    --rttm-channel 1 --rttm-affix "$rttm_affix" \
    $out_dir/xvectors_${name}/cossim_scores $out_dir
  echo "$0: wrote RTTM to output directory ${out_dir}"
fi
