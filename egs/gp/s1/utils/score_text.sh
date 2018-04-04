#!/bin/bash
# Copyright 2010-2011 Microsoft Corporation

if [ -f ./path.sh ]; then . ./path.sh; fi

if [ $# -ne 2 ]; then
   echo "Usage: scripts/score_text.sh <decode-dir> <data-dir>"
   exit 1;
fi

dir=$1
data=$2

if [ ! -f $data/text ]; then
  echo Could not find transcriptions in $data/text
  exit 1
fi


cat $data/text | sed 's:<NOISE>::g' |  sed 's:<SPOKEN_NOISE>::g' > $dir/test_trans.filt

# We assume the transcripts are already in integer form.
cat $dir/*.txt |  sed 's:<UNK>::g' > $dir/text

compute-wer --text --mode=present ark:$dir/test_trans.filt ark,p:$dir/text >& $dir/wer

grep WER $dir/wer

