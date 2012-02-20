#!/bin/bash

# Copyright 2012  Arnab Ghoshal
# Copyright 2010-2011  Microsoft Corporation

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


if [ -f ./path.sh ]; then . ./path.sh; fi

if [ $# -ne 4 ]; then
  echo "Usage: score_text.sh <decode-dir> <word-symbol-table> <data-dir> <phone-map>"
  exit 1;
fi

dir=$1
symtab=$2
data=$3
phonemap=$4

if [ ! -f $data/text ]; then
  echo Could not find transcriptions in $data/text
  exit 1
fi

trans=$data/text
sort -k1,1 $trans > $dir/test.trans

# We assume the transcripts are already in integer form.
cat $dir/*.tra | sort -k1,1 \
  | int2sym.pl --ignore-first-field $symtab \
  | timit_norm_trans.pl -i - -m $phonemap -from 48 -to 39 \
  > $dir/text

compute-wer --text --mode=present ark:$dir/test.trans ark,p:$dir/text \
  >& $dir/wer

grep WER $dir/wer

