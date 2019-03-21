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
# You can use the --num-repeats option to repeat some source data more than
# once when training the LM for the denominator FST.

set -o pipefail

# begin configuration section.
cmd=run.pl
stage=0
num_repeats= # Comma-separated list of positive integer multiplicities, one
             # for each input alignment directory.  The alignments from
             # each source will be scaled by the corresponding value when
             # training the LM.
             # If not specified, weight '1' is used for all data sources.

lm_opts='--num-extra-lm-states=2000'
#end configuration section.


[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -lt 2 ]; then
  echo "Usage: $0 [options] <ali-dir1> [<ali-dir2> ...] <out-dir>";
  echo "e.g.: $0 exp/tri1_ali exp/tri2_ali exp/chain/tdnn_1a_sp";
  echo "Options: "
  echo " --cmd (run.pl|queue.pl...)      # Specify how to run jobs.";
  echo "--lm-opts                        # Options for phone LM generation";
  echo "--num-repeats                    # Comma-separated list of postive integer"
  echo "                                 # multiplicities, one for each input"
  echo "                                 # alignment directory.  The alignments"
  echo "                                 # from each source will be scaled by"
  echo "                                 # the corresponding value when training"
  echo "                                 # the LM.  If not specified, weight '1'"
  echo "                                 # is used for all data sources."
  exit 1;
fi

dir=${@: -1}   # the working directory: last argument to the script
ali_dirs=( $@ )  # read the remaining arguments into an array
unset ali_dirs[${#ali_dirs[@]}-1]  # 'pop' the last argument which is $dir
num_alignments=${#ali_dirs[@]}    # number of alignment dirs to combine

mkdir -p $dir/log
for n in `seq 0 $[$num_alignments-1]`;do
  ali_dir=${ali_dirs[$n]}
  for f in $ali_dir/ali.1.gz $ali_dir/final.mdl $ali_dir/tree; do
    [ ! -f $f ] && echo "$0: Expected file $f to exist" && exit 1;
  done
  utils/lang/check_phones_compatible.sh ${ali_dirs[0]}/phones.txt \
    ${ali_dirs[$n]}/phones.txt || exit 1;
done

cp ${ali_dirs[0]}/tree $dir/ || exit 1

if [ -z "$num_repeats" ]; then
  # If 'num_repeats' is not specified, set num_repeats_array to e.g. (1 1 1).
  num_repeats_array=( $(for n in $(seq $num_alignments); do echo 1; done) )
else
  num_repeats_array=(${num_repeats//,/ })
  num_repeats=${#num_repeats_array[@]}
  if [ $num_repeats -ne $num_alignments ]; then
    echo "$0: too many or too few elements in --num-repeats option: '$num_repeats'"
    exit 1
  fi
fi

all_phones=""  # will contain the names of the .gz files containing phones,
               # with some members possibly repeated per the --num-repeats
               # option
for n in `seq 0 $[num_alignments-1]`; do
  this_num_repeats=${num_repeats_array[$n]}
  this_alignment_dir=${ali_dirs[$n]}
  num_jobs=$(cat $this_alignment_dir/num_jobs)
  if ! [ "$this_num_repeats" -ge 0 ]; then
    echo "Expected comma-separated list of integers for --num-repeats option, got '$num_repeats'"
    exit 1
  fi


  if [ $stage -le 1 ]; then
    for j in $(seq $num_jobs); do gunzip -c $this_alignment_dir/ali.$j.gz; done | \
      ali-to-phones $this_alignment_dir/final.mdl ark:- "ark:|gzip -c >$dir/phones.$n.gz" || exit 1;
  fi

  if [ ! -s $dir/phones.$n.gz ]; then
    echo "$dir/phones.$n.gz is empty or does not exist"
    exit 1
  fi

  all_phones="$all_phones $(for r in $(seq $this_num_repeats); do echo $dir/phones.$n.gz; done)"
done

if [ $stage -le 2 ]; then
  $cmd $dir/log/make_phone_lm_fst.log \
    gunzip -c $all_phones \| \
    chain-est-phone-lm $lm_opts ark:- $dir/phone_lm.fst || exit 1;
  rm $dir/phones.*.gz
fi

if [ $stage -le 3 ]; then
  copy-transition-model ${ali_dirs[0]}/final.mdl $dir/0.trans_mdl || exit 1;
fi

if [ $stage -le 4 ]; then
  $cmd $dir/log/make_den_fst.log \
    chain-make-den-fst $dir/tree $dir/0.trans_mdl \
    $dir/phone_lm.fst \
    $dir/den.fst $dir/normalization.fst || exit 1
fi

echo "Successfully created {den,normalization}.fst"

exit 0
