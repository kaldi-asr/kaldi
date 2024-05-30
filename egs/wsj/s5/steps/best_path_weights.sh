#!/usr/bin/env bash

# Copyright 2014-17 Vimal Manohar

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


# This script gets from the lattice the best path alignments and frame-level
# posteriors of the pdfs in the best path alignment.
# The output directory has the format of an alignment directory.
# It can optionally read alignments from a directory, in which case,
# the script gets frame-level posteriors of the pdf corresponding to those
# alignments.
# The frame-level posteriors in the form of kaldi vectors and are 
# output in weights.scp.

set -e

# begin configuration section.
cmd=run.pl
stage=-10
acwt=0.1
#end configuration section.

if [ -f ./path.sh ]; then . ./path.sh; fi
. utils/parse_options.sh || exit 1;

if [ $# -ne 3 ] && [ $# -ne 4 ]; then
  cat <<EOF
    Usage: $0 [options] <data-dir> <decode-dir> [<ali-dir>] <out-dir>
      E.g. $0 data/train_unt.seg exp/tri1/decode exp/tri1/best_path
    Options:
      --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes.
EOF
  
  exit 1;
fi

data=$1
decode_dir=$2
dir=${@: -1}  # last argument to the script

ali_dir=$dir
if [ $# -eq 4 ]; then
  ali_dir=$3
fi

mkdir -p $dir

nj=$(cat $decode_dir/num_jobs)
echo $nj > $dir/num_jobs

if [ $stage -le 1 ]; then
  mkdir -p $dir/log
  $cmd JOB=1:$nj $dir/log/best_path.JOB.log \
    lattice-best-path --acoustic-scale=$acwt \
      "ark,s,cs:gunzip -c $decode_dir/lat.JOB.gz |" \
      ark:/dev/null "ark:| gzip -c > $dir/ali.JOB.gz" || exit 1
fi

# Find where the final.mdl is.
if [ -f $(dirname $decode_dir)/final.mdl ]; then
  src_dir=$(dirname $decode_dir)
else
  src_dir=$decode_dir
fi

cp $src_dir/cmvn_opts $dir/ || exit 1
for f in final.mat splice_opts frame_subsampling_factor; do
  if [ -f $src_dir/$f ]; then cp $src_dir/$f $dir; fi
done

# make $dir an absolute pathname.
fdir=$(perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $dir ${PWD})

model=$src_dir/final.mdl
tree=$src_dir/tree

for f in $model $decode_dir/lat.1.gz $tree; do
  if [ ! -f $f ]; then echo "$0: expecting file $f to exist" && exit 1; fi
done

cp $model $tree $dir || exit 1

ali_nj=$(cat $ali_dir/num_jobs) || exit 1
if [ $nj -ne $ali_nj ]; then
  echo "$0: $decode_dir and $ali_dir have different number of jobs. Redo alignment with $nj jobs."
  exit 1
fi

if [ $stage -lt 2 ]; then
  $cmd JOB=1:$nj $dir/log/get_post.JOB.log \
    lattice-to-post --acoustic-scale=$acwt \
      "ark,s,cs:gunzip -c $decode_dir/lat.JOB.gz|" ark:- \| \
    post-to-pdf-post $model ark,s,cs:- ark:- \| \
    get-post-on-ali ark,s,cs:- \
    "ark,s,cs:gunzip -c $ali_dir/ali.JOB.gz | convert-ali $dir/final.mdl $model $tree ark,s,cs:- ark:- | ali-to-pdf $model ark,s,cs:- ark:- |" \
    "ark,scp:$fdir/weights.JOB.ark,$fdir/weights.JOB.scp" || exit 1
fi

for n in `seq $nj`; do
  cat $dir/weights.$n.scp 
done > $dir/weights.scp

rm $dir/weights.*.scp

exit 0
