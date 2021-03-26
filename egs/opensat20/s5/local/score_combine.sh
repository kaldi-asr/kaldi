#!/bin/bash

# Copyright 2012-2013  Arnab Ghoshal
#                      Johns Hopkins University (authors: Daniel Povey, Sanjeev Khudanpur)
#                2019  Yiming Wang

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


# Script for system combination using minimum Bayes risk decoding.
# This calls lattice-combine to create a union of lattices that have been
# normalized by removing the total forward cost from them. The resulting lattice
# is used as input to lattice-mbr-decode. This should not be put in steps/ or
# utils/ since the scores on the combined lattice must not be scaled.

# begin configuration section.
cmd=run.pl
beam=6 # prune the lattices prior to MBR decoding, for speed.
stage=0
decode_mbr=true
lat_weights=
word_ins_penalty=0.0
min_lmwt=7
max_lmwt=17
parallel_opts="-pe smp 3"
skip_scoring=false
ctm_name=
#end configuration section.

help_message="Usage: "$(basename $0)" [options] <data-dir> <graph-dir|lang-dir> <decode-dir1> <decode-dir2> [<decode-dir3>... ] <out-dir>
     E.g. "$(basename $0)" data/test data/lang exp/tri1/decode exp/tri2/decode exp/tri3/decode exp/combine
Options:
  --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes.
  --min-lmwt INT                  # minumum LM-weight for lattice rescoring
  --max-lmwt INT                  # maximum LM-weight for lattice rescoring
  --lat-weights STR               # colon-separated string of lattice weights
  --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes.
  --stage (0|1|2)                 # (createCTM | filterCTM | runSclite).
  --parallel-opts <string>        # extra options to command for combination stage,
                                  # default '-pe smp 3'
";

. ./path.sh
. parse_options.sh || exit 1;


if [ $# -lt 5 ]; then
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

for f in $lang/words.txt $lang/phones/word_boundary.int ; do
  [ ! -f $f ] && echo "$0: file $f does not exist" && exit 1;
done


mkdir -p $dir/log
symtab=$lang/words.txt
for i in `seq 0 $[num_sys-1]`; do
  decode_dir=${decode_dirs[$i]}

  model=`dirname $decode_dir`/final.mdl  # model one level up from decode dir
  for f in $model $decode_dir/lat.1.gz ; do
    [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
  done
  if [ $i -eq 0 ]; then
    nj=`cat $decode_dir/num_jobs` || exit 1;
  else
    if [ $nj != `cat $decode_dir/num_jobs` ]; then
      echo "$0: number of decoding jobs mismatches, $nj versus `cat $decode_dir/num_jobs`"
      exit 1;
    fi
  fi
  file_list=""
  # I want to get the files in the correct order so we can use ",s,cs" to avoid
  # memory blowup.  I first tried a pattern like file.{1,2,3,4}.gz, but if the
  # system default shell is not bash (e.g. dash, in debian) this will not work,
  # so we enumerate all the input files.  This tends to make the command lines
  # very long.
  for j in `seq $nj`; do file_list="$file_list $decode_dir/lat.$j.gz"; done

  lats[$i]="ark,s,cs:lattice-scale --inv-acoustic-scale=LMWT 'ark:gunzip -c $file_list|' ark:- | \
 lattice-add-penalty --word-ins-penalty=$word_ins_penalty ark:- ark:- | \
 lattice-align-words $lang/phones/word_boundary.int $model ark:- ark:- |"
done

mkdir -p $dir/scoring/log
local/wer_output_filter < $data/text > $dir/scoring/test_filt.txt
hyp_filtering_cmd="local/wer_output_filter"
if [ -z "$lat_weights" ]; then
  lat_weights=1.0
  for i in `seq $[$num_sys-1]`; do lat_weights="$lat_weights:1.0"; done
fi

if [ -z "$lat_weights" ]; then
  $cmd $parallel_opts LMWT=$min_lmwt:$max_lmwt $dir/log/combine_lats.LMWT.log \
    mkdir -p $dir/score_LMWT/ '&&' \
    lattice-combine "${lats[@]}" ark:- \| \
    lattice-mbr-decode --word-symbol-table=$symtab ark:- \
    ark,t:$dir/scoring/LMWT.tra || exit 1;
else
  $cmd $parallel_opts LMWT=$min_lmwt:$max_lmwt $dir/log/combine_lats.LMWT.log \
    mkdir -p $dir/score_LMWT/ '&&' \
    lattice-combine "${lats[@]}" ark:- \| \
    lattice-mbr-decode --word-symbol-table=$symtab ark:- \
    ark,t:$dir/scoring/LMWT.tra || exit 1;
fi

echo 'scoring'
$cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.log \
  cat $dir/scoring/LMWT.tra \| \
  utils/int2sym.pl -f 2- $symtab \| \
  $hyp_filtering_cmd \| \
  compute-wer --text --mode=present \
  ark:$dir/scoring/test_filt.txt  ark,p:- ">&" $dir/wer_LMWT || exit 1;

cat $dir/scoring/7.tra | utils/int2sym.pl -f 2- $symtab | $hyp_filtering_cmd > $dir/scoring/7.txt || exit 1;
cat $dir/scoring/8.tra | utils/int2sym.pl -f 2- $symtab | $hyp_filtering_cmd > $dir/scoring/8.txt || exit 1;
cat $dir/scoring/9.tra | utils/int2sym.pl -f 2- $symtab | $hyp_filtering_cmd > $dir/scoring/9.txt || exit 1;
cat $dir/scoring/10.tra | utils/int2sym.pl -f 2- $symtab | $hyp_filtering_cmd > $dir/scoring/10.txt || exit 1;
cat $dir/scoring/11.tra | utils/int2sym.pl -f 2- $symtab | $hyp_filtering_cmd > $dir/scoring/11.txt || exit 1;
cat $dir/scoring/12.tra | utils/int2sym.pl -f 2- $symtab | $hyp_filtering_cmd > $dir/scoring/12.txt || exit 1;
cat $dir/scoring/13.tra | utils/int2sym.pl -f 2- $symtab | $hyp_filtering_cmd > $dir/scoring/13.txt || exit 1;
cat $dir/scoring/14.tra | utils/int2sym.pl -f 2- $symtab | $hyp_filtering_cmd > $dir/scoring/14.txt || exit 1;
cat $dir/scoring/15.tra | utils/int2sym.pl -f 2- $symtab | $hyp_filtering_cmd > $dir/scoring/15.txt || exit 1;
cat $dir/scoring/16.tra | utils/int2sym.pl -f 2- $symtab | $hyp_filtering_cmd > $dir/scoring/16.txt || exit 1;
cat $dir/scoring/17.tra | utils/int2sym.pl -f 2- $symtab | $hyp_filtering_cmd > $dir/scoring/17.txt || exit 1;
exit 0
