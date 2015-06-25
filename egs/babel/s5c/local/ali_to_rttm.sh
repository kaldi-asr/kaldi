#!/bin/bash
# Copyright 2012-2013  Johns Hopkins University (Author: Guoguo Chen, Yenda Trmal)
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

#This script will take the ali directory andcreate the corresponding rttm file
#Example
#steps/align_sgmm2.sh --nj 20 --cmd "$decode_cmd" \
#  --transform-dir exp/tri5/decode_dev2h.uem \
#  data/dev2h.uem data/lang exp/sgmm5 exp/sgmm5/align_dev2h.uem
#local/ali_to_rttm.sh data/dev2h data/lang exp/sgmm5/align_dev2h/

cmd=run.pl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40
boost_silence=1.0

if [ -f path.sh ]; then . path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "This script takes an ali directory and creates the corresponding RTTM file"
  echo ""
  echo "Usage: align_text.sh <data-dir> <lang-dir> <exp-dir>"
  echo " e.g.: align_text.sh data/heldout data/lang exp/heldout_ali"
  echo "main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) "

  exit 1;
fi

set -e 
set -o pipefail
set -u

data=$1
lang=$2
dir=$3

oov=`cat $lang/oov.txt`
mkdir -p $dir/log

echo "$0: writing alignments."
wbegin=`grep "#1" $lang/phones.txt | head -1 | awk '{print $2}'`
wend=`grep "#2" $lang/phones.txt | head -1 | awk '{print $2}'`

if [ ! -f $lang/L_align.fst ]; then
  echo "$0: generating $lang/L_align.fst"
  local/make_L_align.sh data/local/tmp.lang/ $lang $lang 2>&1 | tee $dir/log/L_align.log
fi

$cmd $dir/log/align_to_words.log \
  ali-to-phones $dir/final.mdl "ark:gunzip -c $dir/ali.*.gz|" ark,t:- \| \
  phones-to-prons $lang/L_align.fst $wbegin $wend ark:- "ark,s:utils/sym2int.pl -f 2- --map-oov '$oov' $lang/words.txt <$data/text|" ark,t:- \| \
  prons-to-wordali ark:- "ark:ali-to-phones --write-lengths=true $dir/final.mdl 'ark:gunzip -c $dir/ali.*.gz|' ark,t:- |" ark,t:$dir/align.txt 

echo "$0: done writing alignments."

echo "$0: writing rttm."
[ ! -x local/txt_to_rttm.pl ] && \
  echo "Not creating rttm because local/txt2rttm.pl does not exist or not executable." && exit 1;

local/txt_to_rttm.pl --symtab=$lang/words.txt --segment=$data/segments $dir/align.txt $dir/rttm 2>$dir/log/rttm.log
echo "$0: done writing rttm."

exit 0;
