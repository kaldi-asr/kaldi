#!/bin/bash
# Copyright 2013-2018  Johns Hopkins University (authors: Guoguo Chen, Yenda Trmal)

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


set -o pipefail
set -e
set -x

if [ $# -ne 3 ]; then
  echo "This is a simple script that will generate the L_align.fst"
  echo "The FST L_align.fst is used for getting the force-aligned "
  echo "utterances"
  echo "The script automaticky recognizes the probabilistic lexicon"
  echo "is used and will use the correct file"
  echo ""
  echo "usage: local/L_align.sh <lang-local-dir> <lang-dir> <out-dir>"
  echo "e.g.: local/L_align.sh data/local/lang data/lang data/lang"
  exit 1;
fi

tmpdir=$1
dir=$2
outdir=$3

silphone=`cat $dir/phones/optional_silence.txt` || exit 1;

# Create lexicon with alignment info
if  [ -f $tmpdir/lexicon.txt ] ; then
  cat $tmpdir/lexicon.txt | \
    awk '{printf("%s #1 ", $1); for (n=2; n <= NF; n++) { printf("%s ", $n); } print "#2"; }' | \
    utils/make_lexicon_fst.pl - 0.5 $silphone | \
    fstcompile --isymbols=$dir/phones.txt --osymbols=$dir/words.txt \
    --keep_isymbols=false --keep_osymbols=false | \
    fstarcsort --sort_type=olabel > $outdir/L_align.fst
elif [ -f $tmpdir/lexiconp.txt ] ;  then
  cat $tmpdir/lexiconp.txt | \
    awk '{printf("%s #1 ", $1); for (n=3; n <= NF; n++) { printf("%s ", $n); } print "#2"; }' | \
    utils/make_lexicon_fst.pl - 0.5 $silphone | \
    fstcompile --isymbols=$dir/phones.txt --osymbols=$dir/words.txt \
    --keep_isymbols=false --keep_osymbols=false | \
    fstarcsort --sort_type=olabel > $outdir/L_align.fst
else
  echo >&2 "Neither $tmpdir/lexicon.txt nor $tmpdir/lexiconp.txt does not exist"
  exit 1
fi
exit 0;
