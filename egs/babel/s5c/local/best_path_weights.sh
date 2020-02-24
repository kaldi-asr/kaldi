#!/usr/bin/env bash

# Copyright 2014 Vimal Manohar

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


# This script combines frame-level posteriors from different decode
# directories. The first decode directory is assumed to be the primary
# and is used to get the best path. The posteriors from other decode
# directories are interpolated with the posteriors of the best path.
# The output is a new directory with final.mdl, tree from the primary
# decode-dir and the best path alignments and weights in a decode-directory
# with the same basename as the primary directory.
# This is typically used to get better posteriors for semisupervised training
# of DNN
# e.g. local/combine_posteriors.sh exp/tri6_nnet/decode_train_unt.seg
# exp/sgmm_mmi_b0.1/decode_fmllr_train_unt.seg_it4 exp/combine_dnn_sgmm
# Here the final.mdl and tree are copied from exp/tri6_nnet to
# exp/combine_dnn_sgmm. best_path_ali.*.gz obtained from the primary dir and
# the interpolated posteriors in weights.*.gz are placed in
# exp/combine_dnn_sgmm/decode_train_unt.seg

set -e

# begin configuration section.
cmd=run.pl
stage=-10
#end configuration section.

help_message="Usage: "$(basename $0)" [options] <data-dir> <graph-dir|lang-dir> <decode-dir1>[:weight] <decode-dir2>[:weight] [<decode-dir3>[:weight] ... ] <out-dir>
     E.g. "$(basename $0)" data/train_unt.seg data/lang exp/tri1/decode:0.5 exp/tri2/decode:0.25 exp/tri3/decode:0.25 exp/combine
Options:
  --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes.
";

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;


if [ $# -lt 4 ]; then
  printf "$help_message\n";
  exit 1;
fi

data=$1
lang=$2
dir=${@: -1}  # last argument to the script
shift 2;
decode_dirs=( $@ )  # read the remaining arguments into an array
unset decode_dirs[${#decode_dirs[@]}-1]  # 'pop' the last argument which is odir
num_sys=${#decode_dirs[@]}  # number of systems to combine

mkdir -p $dir
mkdir -p $dir/log

decode_dir=`echo ${decode_dirs[0]} | cut -d: -f1`
nj=`cat $decode_dir/num_jobs`

out_decode=$dir/`basename $decode_dir`
mkdir -p $out_decode

if [ $stage -lt -1 ]; then
  mkdir -p $out_decode/log
  $cmd JOB=1:$nj $out_decode/log/best_path.JOB.log \
    lattice-best-path --acoustic-scale=0.1 \
    "ark,s,cs:gunzip -c $decode_dir/lat.JOB.gz |" \
    ark:/dev/null "ark:| gzip -c > $out_decode/best_path_ali.JOB.gz" || exit 1
fi

weights_sum=0.0

for i in `seq 0 $[num_sys-1]`; do
  decode_dir=${decode_dirs[$i]}

  weight=`echo $decode_dir | cut -d: -s -f2`
  [ -z "$weight" ] && weight=1.0

  if [ $i -eq 0 ]; then
    file_list="\"ark,s,cs:gunzip -c $out_decode/weights.$i.JOB.gz | vector-scale --scale=$weight ark:- ark:- |\""
  else
    file_list="$file_list \"ark,s,cs:gunzip -c $out_decode/weights.$i.JOB.gz | vector-scale --scale=$weight ark:- ark:- |\""
  fi

  weights_sum=`perl -e "print STDOUT $weights_sum + $weight"`
done

inv_weights_sum=`perl -e "print STDOUT 1.0/$weights_sum"`

for i in `seq 0 $[num_sys-1]`; do
  if [ $stage -lt $i ]; then
    decode_dir=`echo ${decode_dirs[$i]} | cut -d: -f1`

    model=`dirname $decode_dir`/final.mdl  # model one level up from decode dir
    tree=`dirname $decode_dir`/tree        # tree one level up from decode dir

    for f in $model $decode_dir/lat.1.gz $tree; do
      [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
    done
    if [ $i -eq 0 ]; then
      nj=`cat $decode_dir/num_jobs` || exit 1;
      cp $model $dir || exit 1
      cp $tree $dir || exit 1
      echo $nj > $out_decode/num_jobs
    else
      if [ $nj != `cat $decode_dir/num_jobs` ]; then
        echo "$0: number of decoding jobs mismatches, $nj versus `cat $decode_dir/num_jobs`"
        exit 1;
      fi
    fi

    $cmd JOB=1:$nj $dir/log/get_post.$i.JOB.log \
      lattice-to-post --acoustic-scale=0.1 \
      "ark,s,cs:gunzip -c $decode_dir/lat.JOB.gz|" ark:- \| \
      post-to-pdf-post $model ark,s,cs:- ark:- \| \
      get-post-on-ali ark,s,cs:- "ark,s,cs:gunzip -c $out_decode/best_path_ali.JOB.gz | convert-ali $dir/final.mdl $model $tree ark,s,cs:- ark:- | ali-to-pdf $model ark,s,cs:- ark:- |" "ark:| gzip -c > $out_decode/weights.$i.JOB.gz" || exit 1
  fi
done

if [ $stage -lt $num_sys ]; then
  if [ "$num_sys" -eq 1 ]; then
    $cmd JOB=1:$nj $dir/log/move_post.JOB.log \
      mv $out_decode/weights.0.JOB.gz $out_decode/weights.JOB.gz || exit 1
  else
    $cmd JOB=1:$nj $dir/log/interpolate_post.JOB.log \
      vector-sum $file_list \
      "ark:| vector-scale --scale=$inv_weights_sum ark:- ark:- | gzip -c > $out_decode/weights.JOB.gz" || exit 1
  fi
fi

exit 0
