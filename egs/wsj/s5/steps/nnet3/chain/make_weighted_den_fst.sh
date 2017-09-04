#!/bin/bash

# Copyright 2017 Vimal Manohar
#           2017 Pegah Ghahremani
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
# chain training. It additionally copies the transition model and tree from the
# first alignment directory to the chain directory.
# This script can accept multiple sources of alignments with same phone sets
# that can be weighted to estimate phone LM.
# 'weights' is comma-separated list of positive int values used
# to scale different phone sequences for different alignments.
# Each alignment directory should contain tree, final.mdl and ali.*.gz.

set -o pipefail

# begin configuration section.
cmd=run.pl
stage=-10
weights= # comma-separated list of positive int valued scale weights used
         # to scale different phone sequences for different alignments.
         # Scaling the count with i^th int weight 'w' is done by repeating
         # the i^th phone sequence 'w' times.
         # i.e. "1,10"
         # If not specified, weight '1' is used for all phone sequences.

lm_opts='--num-extra-lm-states=2000'
#end configuration section.


[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -lt 2 ]; then
  echo "Usage: $0 [options] <ali-dir1> [<ali-dir2> ...] <out-dir>";
  echo "e.g.: $0 exp/tri1_ali exp/tri2_ali exp/chain/tdnn_1a_sp";
  echo "Options: "
  echo " --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes.";
  echo "--lm-opts                        # options for phone LM generation";
  exit 1;
fi

dir=${@: -1}  # last argument to the script
ali_dirs=( $@ )  # read the remaining arguments into an array
unset ali_dirs[${#ali_dirs[@]}-1]  # 'pop' the last argument which is odir
num_alignments=${#ali_dirs[@]}  # number of alignment dirs to combine

mkdir -p $dir/log
for n in `seq 0 $[$num_alignments-1]`;do
  ali_dir=${ali_dirs[$n]}
  for f in $ali_dir/ali.1.gz $ali_dir/final.mdl $ali_dir/tree; do
    if [ ! -f $f ]; then
      echo "$0: Could not find file $f"
      exit 1
    fi
  done
  utils/lang/check_phones_compatible.sh ${ali_dirs[0]}/phones.txt \
    ${ali_dirs[$n]}/phones.txt || exit 1;
done

cp ${ali_dirs[0]}/tree $dir/ || exit 1

if [ -z $weights ]; then
  # If 'weights' is not specified, comma-separated array '1' with dim
  #'num_alignments' is defined as 'weights'.
  for n in `seq 1 $num_alignments`;do weights="$weights,1"; done
else
  w_arr=(${weights//,/ })
  num_weights=${#w_arr[@]}
  if [ $num_alignments -ne $num_weights ]; then
    echo "$0: number of weights in $weight, $num_weights, should be equal to the "
    echo "number of alignment directories, $num_alignments." && exit 1;
  fi
fi

if [ $stage -le 1 ]; then
  for n in `seq 0 $[num_alignments-1]`; do
    w=$(echo $weights | cut -d, -f$[$n+1])
    adir=${ali_dirs[$n]}
    num_jobs=$(cat $adir/num_jobs)
    if ! [[ $w =~ ^[+]?[0-9]+$ ]] ; then
      echo "no positive int weight specified for alignment ${ali_dirs[$n]}" && exit 1;
    fi
    rm $adir/alignment_files.txt 2>/dev/null || true
    for x in `seq $w`;do
      for j in `seq $num_jobs`;do
        echo $adir/ali.$j.gz >> $adir/alignment_files.txt
      done
    done
  done
  $cmd $dir/log/make_phone_lm_fst.log \
    ali_dirs=\(${ali_dirs[@]}\) \; \
    for n in `seq 0 $[num_alignments-1]`\; do \
      adir=\${ali_dirs[\$n]} \; \
      cat \$adir/alignment_files.txt \| while read f\; do gunzip -c \$f \; done \| \
        ali-to-phones \$adir/final.mdl ark:- ark:- \; \
    done \| \
      chain-est-phone-lm $lm_opts ark:- $dir/phone_lm.fst || exit 1;
fi

if [ $stage -le 2 ]; then
  copy-transition-model ${ali_dirs[0]}/final.mdl $dir/0.trans_mdl || exit 1;
fi

if [ $stage -le 3 ]; then
  $cmd $dir/log/make_den_fst.log \
    chain-make-den-fst $dir/tree $dir/0.trans_mdl \
    $dir/phone_lm.fst \
    $dir/den.fst $dir/normalization.fst || exit 1
fi

exit 0
