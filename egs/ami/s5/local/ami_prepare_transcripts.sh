#!/bin/bash

trans=data/local/annotations/transcripts1
[ ! -f $trans1 ]  && echo "$0: File $trans not found. " && exit 1;

wdir=data/local/annotations

local/ami_split_segments.pl $trans $wdir/transcripts2 &> $wdir/log/split_segments.log



