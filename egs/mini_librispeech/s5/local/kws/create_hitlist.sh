#!/usr/bin/env bash
# Copyright 2012-2018  Johns Hopkins University (Author: Guoguo Chen, Yenda Trmal)
# Apache 2.0.

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


cmd=run.pl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40
boost_silence=1.0

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
  echo "This script takes an ali directory and creates the corresponding RTTM file"
  echo ""
  echo "Usage: create_hitlist.sh <data-dir> <lang-dir> <lang-tmp-dir> <exp-dir> <kws-data-dir>"
  echo " e.g.: create_hitlist.sh data/heldout data/lang data/local/lang_tmp exp/heldout_ali data/heldout/kws"
  echo "main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) "

  exit 1;
fi

set -e
set -o pipefail
set -u

data=$1
lang=$2
lang_tmp=$3
dir=$4
kws=$5

oov=`cat $lang/oov.txt`
mkdir -p $dir/log

echo "$0: writing alignments."
wbegin=`grep "#1" $lang/phones.txt | head -1 | awk '{print $2}'`
wend=`grep "#2" $lang/phones.txt | head -1 | awk '{print $2}'`

if [ ! -f $lang/L_align.fst ]; then
  echo "$0: generating $lang/L_align.fst"
  local/kws/make_L_align.sh $lang_tmp $lang $lang 2>&1 | tee $dir/log/L_align.log
fi

$cmd $dir/log/ali_to_hitlist.log \
  set -e -o pipefail\; \
  ali-to-phones $dir/final.mdl "ark:gunzip -c $dir/ali.*.gz|" ark,t:- \| \
  phones-to-prons $lang/L_align.fst $wbegin $wend ark:- "ark,s:utils/sym2int.pl -f 2- --map-oov '$oov' $lang/words.txt <$data/text|" ark,t:- \| \
  prons-to-wordali ark:- "ark:ali-to-phones --write-lengths=true $dir/final.mdl 'ark:gunzip -c $dir/ali.*.gz|' ark,t:- |" ark,t:- \| \
  local/kws/generate_hitlist.pl $kws/keywords.int \|\
  utils/sym2int.pl -f 2 $kws/utt.map  \> $kws/hitlist

echo "$0: done generating hitlist"


exit 0;
