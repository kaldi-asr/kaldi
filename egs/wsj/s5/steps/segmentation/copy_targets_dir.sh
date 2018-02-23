#!/bin/bash

# Copyright    2017  Nagendra Kumar Goel
#              2014  Johns Hopkins University (author: Nagendra K Goel)
# Apache 2.0

# This script makes a copy of targets directory (by copying targets.scp),
# possibly adding a specified prefix or a suffix to the utterance names.

# begin configuration section
utt_prefix=
utt_suffix=
# end configuration section

if [ -f ./path.sh ]; then . ./path.sh; fi
. ./utils/parse_options.sh

if [ $# != 2 ]; then
  echo "Usage: "
  echo "  $0 [options] <srcdir> <destdir>"
  echo "e.g.:"
  echo " $0  --utt-prefix=1- exp/segmentation_1a/train_whole_combined_targets_sub3 exp/segmentation_1a/train_whole_combined_targets_sub3_rev1"
  echo "Options"
  echo "   --utt-prefix=<prefix>     # Prefix for utterance ids, default empty"
  echo "   --utt-suffix=<suffix>     # Suffix for utterance ids, default empty"
  exit 1;
fi

export LC_ALL=C

srcdir=$1
destdir=$2

mkdir -p $destdir

if [ -f $srcdir/frame_subsampling_factor ]; then
  cp $srcdir/frame_subsampling_factor $destdir
fi

cat $srcdir/targets.scp | awk -v p=$utt_prefix -v s=$utt_suffix \
  '{printf("%s %s%s%s\n", $1, p, $1, s);}' > $destdir/utt_map

cat $srcdir/targets.scp | utils/apply_map.pl -f 1 $destdir/utt_map | \
  sort -k1,1 > $destdir/targets.scp

echo "$0: copied targets from $srcdir to $destdir"
