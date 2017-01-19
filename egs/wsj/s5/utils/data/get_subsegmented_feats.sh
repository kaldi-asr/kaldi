#! /bin/bash

# Copyright 2016  Johns Hopkins University (Author: Dan Povey)
#           2016  Vimal Manohar
# Apache 2.0.

if [ $# -ne 4 ]; then
  echo "This scripts gets subsegmented_feats (by adding ranges to data/feats.scp) "
  echo "for the subsegments file. This is does one part of the "
  echo "functionality in subsegment_data_dir.sh, which additionally "
  echo "creates a new subsegmented data directory."
  echo "Usage: $0 <feats> <frame-shift> <frame-overlap> <subsegments>" 
  echo " e.g.: $0 data/train/feats.scp 0.01 0.015 subsegments"
  exit 1
fi

feats=$1
frame_shift=$2
frame_overlap=$3
subsegments=$4

# The subsegments format is <new-utt-id> <old-utt-id> <start-time> <end-time>.
# e.g. 'utt_foo-1 utt_foo 7.21 8.93'
# The first awk command replaces this with the format:
# <new-utt-id> <old-utt-id> <first-frame> <last-frame>
# e.g. 'utt_foo-1 utt_foo 721 893'
# and the apply_map.pl command replaces 'utt_foo' (the 2nd field) with its corresponding entry
# from the original wav.scp, so we get a line like:
# e.g. 'utt_foo-1 foo-bar.ark:514231 721 892'
# Note: the reason we subtract one from the last time is that it's going to
# represent the 'last' frame, not the 'end' frame [i.e. not one past the last],
# in the matlab-like, but zero-indexed [first:last] notion.  For instance, a segment with 1 frame
# would have start-time 0.00 and end-time 0.01, which would become the frame range
# [0:0]
# The second awk command turns this into something like
# utt_foo-1 foo-bar.ark:514231[721:892]
# It has to be a bit careful because the format actually allows for more general things
# like pipes that might contain spaces, so it has to be able to produce output like the
# following:
# utt_foo-1 some command|[721:892]
# Lastly, utils/data/normalize_data_range.pl will only do something nontrivial if
# the original data-dir already had data-ranges in square brackets.
awk -v s=$frame_shift -v fovlp=$frame_overlap '{print $1, $2, int(($3/s)+0.5), int(($4-fovlp)/s+0.5);}' <$subsegments| \
  utils/apply_map.pl -f 2 $feats | \
  awk '{p=NF-1; for (n=1;n<NF-2;n++) printf("%s ", $n); k=NF-2; l=NF-1; printf("%s[%d:%d]\n", $k, $l, $NF)}' | \
  utils/data/normalize_data_range.pl  
