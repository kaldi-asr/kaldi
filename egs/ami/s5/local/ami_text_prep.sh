#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <ami-dir>"
  exit 1;
fi

amidir=$1
wdir=data/local/annotations

#extract text from AMI XML annotations
local/ami_xml2text.sh $amidir

[ ! -f $wdir/transcripts1 ] && echo "$0: File $wdir/transcripts1 not found." && exit 1;

echo "Preprocessing transcripts..."
local/ami_split_segments.pl $wdir/transcripts1 $wdir/transcripts2 &> $wdir/log/split_segments.log


#HMM
#MM HMM
#MM UHM


grep -f local/split_train.orig $wdir/transcripts2 > $wdir/train.txt
grep -f local/split_dev.orig $wdir/transcripts2 > $wdir/dev.txt
grep -f local/split_eval.orig $wdir/transcripts2 > $wdir/eval.txt







