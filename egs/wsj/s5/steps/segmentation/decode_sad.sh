#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

# This script does Viterbi decoding using a matrix of frame log-likelihoods 
# with the columns corresponding to the pdfs.
# It is a wrapper around the binary decode-faster-mapped.

set -e
set -o pipefail

cmd=run.pl
acwt=0.1
beam=8
max_active=1000
likes_prefix=log_likes    # prefix of the archives to read from.
                          # e.g. read from log_likes.*.gz
transform=   # Transformation matrix to apply on the input archives read from {likes_prefix}.*.gz 
apply_log=false    # If true, the log is applied on the transformed input matrix. Applicable when input is probabilities.
priors=   # A vector of counts, which will be used to subtract the log-priors 
          # before passing to the decoder

. path.sh

. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 <graph-dir> <log_likes_dir> <decode-dir>"
  echo " e.g.: $0 "
  exit 1 
fi

graph_dir=$1
log_likes_dir=$2
dir=$3

mkdir -p $dir/log

nj=`cat $log_likes_dir/num_jobs`
echo $nj > $dir/num_jobs

extra_files=$log_likes_dir/$likes_prefix.1.gz
for f in $graph_dir/HCLG.fst $extra_files; do
  if [ ! -f $f ]; then
    echo "$0: Could not find file $f"
    exit 1
  fi
done

rspecifier="ark:gunzip -c $log_likes_dir/$likes_prefix.JOB.gz |"

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
