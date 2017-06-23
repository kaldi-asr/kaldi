#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

# This script does Viterbi decoding using a matrix of frame log-likelihoods 
# with the columns corresponding to the pdfs.
# It is a wrapper around the binary decode-faster-mapped.

set -e
set -o pipefail

cmd=run.pl
nj=4
acwt=0.1
beam=8
max_active=1000
transform=   # Transformation matrix to apply on the input archives read from output.scp
apply_log=false    # If true, the log is applied on the transformed input matrix. Applicable when input is probabilities.
priors=   # A vector of counts, which will be used to subtract the log-priors 
          # before passing to the decoder

. path.sh

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

# Apply a transformation on the input matrix to combine scores from different columns
if [ ! -z "$transform" ]; then
  rspecifier="$rspecifier transform-feats $transform ark:- ark:- |"
fi

if $apply_log; then
  rspecifier="$rspecifier copy-matrix --apply-log ark:- ark:- |"
fi

# Subtract log-priors to convert log-odds to pseudo log-likelihoods for decoding.
if [ ! -z $priors ]; then
  {
  copy-vector --binary=false $priors - | \
    awk '{ for (i = 2; i < NF; i++) { sum += $i; };
  printf ("[");
  for (i = 2; i < NF; i++) { printf " "log($i/sum); };
  print (" ]"); }' > $dir/log_priors.vec;
  } 2> $dir/log/get_log_priors.log || exit 1
  if [ ! -f $dir/log_priors.vec ]; then
    echo "$0: Did not create $dir/log_priors.vec"
    exit 1 
  fi

  rspecifier="$rspecifier matrix-add-offset ark:- 'vector-scale --scale=-1.0 $dir/log_priors.vec - |' ark:- |"
fi

decoder_opts+=(--acoustic-scale=$acwt --beam=$beam --max-active=$max_active)

$cmd JOB=1:$nj $dir/log/decode.JOB.log \
  decode-faster ${decoder_opts[@]} \
  $graph_dir/HCLG.fst "$rspecifier" \
  ark:/dev/null "ark:| gzip -c > $dir/ali.JOB.gz"
