#!/usr/bin/env bash
# Copyright 2013  Johns Hopkins University (authors: Yenda Trmal)
#           2018  Vimal Manohar

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

# This scoring script is copied from Babel and modified.
# This is a scoring script for the CTMS in <decode-dir>/score_<LMWT>/${name}.ctm
# it tries to mimic the NIST scoring setup as much as possible (and usually does a good job)

# begin configuration section.
cmd=run.pl
cer=0
min_lmwt=7
max_lmwt=17
model=
stage=0
ctm_name=
word_ins_penalty=0.0,0.5,1.0
case_insensitive=true
use_icu=true
icu_transform='Any-Lower'
#end configuration section.

echo $0 $@

[ -f ./path.sh ] && . ./path.sh
[ -f ./cmd.sh ]  && . ./cmd.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <dataDir> <langDir|graphDir> <decodeDir>" && exit;
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --cer (0|1)                     # compute CER in addition to WER"
  exit 1;
fi

data=$1
lang=$2 # This parameter is not used -- kept only for backwards compatibility
dir=$3

set -e
set -o pipefail
set -u

ScoringProgram=`which sclite` || ScoringProgram=$KALDI_ROOT/tools/sctk/bin/sclite
[ ! -x $ScoringProgram ] && echo "Cannot find scoring program at $ScoringProgram" && exit 1;
SortingProgram=`which hubscr.pl` || SortingProgram=$KALDI_ROOT/tools/sctk/bin/hubscr.pl
[ ! -x $ScoringProgram ] && echo "Cannot find scoring program at $ScoringProgram" && exit 1;

stm_filter_cmd=cat
[ -x local/stm_filter ] && stm_filter_cmd=local/stm_filter
ctm_filter_cmd=cat
[ -x local/ctm_filter ] && ctm_filter_cmd=local/ctm_filter

for f in $data/stm  ; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

if [ -z $ctm_name ] ; then
  name=`basename $data`; # e.g. eval2000
else
  name=$ctm_name
fi

if [ $stage -le 0 ] ; then
  for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
    mkdir -p $dir/scoring/penalty_$wip/log
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/penalty_$wip/log/score.LMWT.log \
      set -e';' set -o pipefail';' \
      cat $dir/score_LMWT_${wip}/${name}.ctm \| $ctm_filter_cmd '>' $dir/score_LMWT_${wip}/${name}.ctm.unsorted '&&' \
      cat $data/stm \| $stm_filter_cmd '>' $dir/score_LMWT_${wip}/stm.unsorted '&&' \
      $SortingProgram sortSTM \<$dir/score_LMWT_${wip}/stm.unsorted          \>$dir/score_LMWT_${wip}/stm.sorted '&&' \
      $SortingProgram sortCTM \<$dir/score_LMWT_${wip}/${name}.ctm.unsorted  \>$dir/score_LMWT_${wip}/${name}.ctm.sorted '&&' \
      paste -d ' ' \<\(cut -f 1-5 -d ' ' $dir/score_LMWT_${wip}/stm.sorted \) \
                   \<\(cut -f 6- -d ' ' $dir/score_LMWT_${wip}/stm.sorted \| uconv -f utf8 -t utf8 -x "$icu_transform" \) \
          \> $dir/score_LMWT_${wip}/stm '&&' \
      paste -d ' ' \<\(cut -f 1-4 -d ' ' $dir/score_LMWT_${wip}/${name}.ctm.sorted \) \
                   \<\(cut -f 5-  -d ' ' $dir/score_LMWT_${wip}/${name}.ctm.sorted \| uconv -f utf8 -t utf8 -x "$icu_transform" \) \
          \> $dir/score_LMWT_${wip}/${name}.ctm.sorted2 '&&' \
      utils/fix_ctm.sh $dir/score_LMWT_${wip}/stm $dir/score_LMWT_${wip}/${name}.ctm.sorted2 '&&' \
      $SortingProgram sortCTM \<$dir/score_LMWT_${wip}/${name}.ctm.sorted2  \>$dir/score_LMWT_${wip}/${name}.ctm '&&' \
      $ScoringProgram -s -r $dir/score_LMWT_${wip}/stm  stm -h $dir/score_LMWT_${wip}/${name}.ctm ctm \
        -n "$name.ctm" -f 0 -D -F  -o  sum rsum prf dtl sgml -e utf-8 || exit 1
  done
fi

if [ $stage -le 1 ]; then
  if [ $cer -eq 1 ]; then
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/penalty_$wip/log/score.LMWT.char.log \
      $ScoringProgram -s -r $dir/score_LMWT_${wip}/stm stm -h $dir/score_LMWT_${wip}/${name}.ctm ctm \
        -n "$name.char.ctm" -o sum rsum prf dtl sgml -f 0 -D -F -c NOASCII DH -e utf-8 || exit 1
  fi
fi


echo "Finished scoring on" `date`
exit 0
