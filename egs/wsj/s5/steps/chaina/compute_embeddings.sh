#!/bin/bash

# Copyright 2012-2019  Johns Hopkins University (Author: Daniel Povey).
#                2016  Vimal Manohar
# Apache 2.0.

# This script is used to compute the embeddings (the output of
# 'bottom.raw' and dump them to disk.

# Begin configuration section.
stage=1
nj=4 # number of jobs.
cmd=run.pl
frames_per_chunk=50
extra_left_context=0
extra_right_context=0
compress=true     # Specifies whether the output should be compressed before
                  # dumping to disk
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;
set -e -u

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <data-dir> <model-subdirectory> <output-dir>"
  echo "e.g.:   steps/chaina/compute_embeddings.sh --nj 8 \\"
  echo "    data/test_eval92_hires exp/chaina/tdnn1_sp/final exp/nnet3/tdnn1_sp/data/final/test_eval92_hires"
  echo "Output will be in <output-dir>/output.scp"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                   # config containing options"
  echo "  --nj <nj>                                # number of parallel jobs"
  echo "  --cmd <cmd>                              # Command to run in parallel with"
  echo "  --iter <iter>                            # Iteration of model to decode; default is final."
  exit 1;
fi

data=$1
model_dir=$2
dir=$3

mkdir -p $dir/log

# convert $dir to absolute pathname
fdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $dir ${PWD}`

for f in $model_dir/bottom.raw $model_dir/info.txt $data/feats.scp; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f"
    exit 1
  fi
done


sdata=$data/split$nj;
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs



bottom_subsampling_factor=$(awk '/^bottom_subsampling_factor/ {print $2}' <$model_dir/info.txt)
if ! [ $bottom_subsampling_factor -gt 0 ]; then
  echo "$0: error getting bottom_subsampling_factor from $model_dir/info.txt"
  exit 1
fi



if [ $stage -le 1 ]; then
  $cmd JOB=1:$nj $dir/log/compute.JOB.log \
    nnet3-compute --use-gpu=no \
      --frame-subsampling-factor=$bottom_subsampling_factor \
      --frames-per-chunk=$frames_per_chunk \
      --extra-left-context=$extra_left_context \
      --extra-right-context=$extra_right_context \
      $model_dir/bottom.raw scp:$sdata/JOB/feats.scp \
      "ark:|copy-feats --compress=$compress ark:- ark,scp:$dir/output.JOB.ark,$dir/output.JOB.scp"
fi

for n in $(seq $nj); do
  cat $dir/output.$n.scp
done > $dir/output.scp

exit 0;
