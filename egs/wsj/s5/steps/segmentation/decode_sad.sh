#!/usr/bin/env bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

# This script does Viterbi decoding using a matrix of frame log-likelihoods 
# with the columns corresponding to the pdfs.
# It is a wrapper around the binary decode-faster.

set -e
set -o pipefail

cmd=run.pl
nj=4
acwt=0.1
beam=8
max_active=1000
transform=   # Transformation matrix to apply on the input archives read from output.scp

. ./path.sh

. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 <graph-dir> <nnet_output_dir> <decode-dir>"
  echo " e.g.: $0 "
  exit 1 
fi

graph_dir=$1
nnet_output_dir=$2
dir=$3

mkdir -p $dir/log

echo $nj > $dir/num_jobs

for f in $graph_dir/HCLG.fst $nnet_output_dir/output.scp $extra_files; do
  if [ ! -f $f ]; then
    echo "$0: Could not find file $f"
    exit 1
  fi
done

rspecifier="ark:utils/split_scp.pl -j $nj \$[JOB-1] $nnet_output_dir/output.scp | copy-feats scp:- ark:- |"

# Apply a transformation on the input matrix to combine 
# probs from different columns to pseudo-likelihoods
if [ ! -z "$transform" ]; then
  rspecifier="$rspecifier transform-feats $transform ark:- ark:- |"
fi

# Convert pseudo-likelihoods to pseudo log-likelihood
rspecifier="$rspecifier copy-matrix --apply-log ark:- ark:- |"

decoder_opts+=(--acoustic-scale=$acwt --beam=$beam --max-active=$max_active)

$cmd JOB=1:$nj $dir/log/decode.JOB.log \
  decode-faster ${decoder_opts[@]} \
  $graph_dir/HCLG.fst "$rspecifier" \
  ark:/dev/null "ark:| gzip -c > $dir/ali.JOB.gz"
