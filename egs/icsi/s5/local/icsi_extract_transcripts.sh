#!/bin/bash

meetlist=$1
transdir=$2
outdir=$3

mkdir -p $outdir

while read line; do
  echo "Parsing $line"
  local/icsi_parse_transcripts.pl $transdir/$line.mrt $outdir/$line.txt
done < $meetlist;

