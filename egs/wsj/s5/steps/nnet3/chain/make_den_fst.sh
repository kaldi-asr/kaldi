#!/bin/bash

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

# This script creates denominator FST (den.fst) and normalization.fst for 
# chain training. It additional copies the transition model and tree from the 
# first alignment directory to the chain directory.
# This script can accept multiple sources of alignments that can be 
# weighted to estimate phone LM.

set -o pipefail

# begin configuration section.
cmd=run.pl
stage=-10
weights=
#end configuration section.

help_message="Usage: "$(basename $0)" [options] <ali-dir1> [<ali-dir2> ...] <out-dir>
     E.g. "$(basename $0)" exp/tri1_ali exp/tri2_ali exp/chain/tdnn_1a_sp
Options:
  --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes.
";

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -lt 2 ]; then
  printf "$help_message\n";
  exit 1;
fi

dir=${@: -1}  # last argument to the script
ali_dirs=( $@ )  # read the remaining arguments into an array
unset ali_dirs[${#ali_dirs[@]}-1]  # 'pop' the last argument which is odir
num_sys=${#ali_dirs[@]}  # number of systems to combine

mkdir -p $dir/log

ali_dir=`echo ${ali_dirs[0]} | cut -d: -f1`

for f in $ali_dir/ali.1.gz $ali_dir/final.mdl $ali_dir/tree; do
  if [ ! -f $f ]; then
    echo "$0: Could not find file $f"
    exit 1
  fi
done

cp $ali_dir/tree $dir/ || exit 1

for n in `seq 0 $[num_sys-1]`; do
  adir=${ali_dirs[$n]}
  alignments+=("ark:gunzip -c $adir/ali.*.gz | ali-to-phones $adir/final.mdl ark:- ark:- |")
done

if [ $stage -le 1 ]; then
  $cmd $dir/log/make_phone_lm.log \
    chain-est-phone-lm $lm_opts --scales="$weights" \
    "${alignments[@]}" $dir/phone_lm.fst || exit 1
fi 

if [ $stage -le 2 ]; then
  copy-transition-model $ali_dir/final.mdl $dir/0.trans_mdl
fi

if [ $stage -le 3 ]; then
  $cmd $dir/log/make_den_fst.log \
    chain-make-den-fst $dir/tree $dir/0.trans_mdl \
    $dir/phone_lm.fst \
    $dir/den.fst $dir/normalization.fst || exit 1
fi

exit 0
