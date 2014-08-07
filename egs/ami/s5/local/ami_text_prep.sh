#!/bin/bash

# Copyright 2014, University of Edinburgh (Author: Pawel Swietojanski), 2014, Apache 2.0

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

#make final train/dev/eval splits
for dset in train eval dev; do
  [ ! -f local/split_$dset.final  ] &&  cp local/split_$dset.orig local/split_$dset.final
  grep -f local/split_$dset.final $wdir/transcripts2 > $wdir/$dset.txt
done







