#!/bin/bash

# Copyright 2016    Vimal Manohar
#           2017    Hossein Hadian
# Apache 2.0

echo "$0 $@"  # Print the command line for logging
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo " Usage: $0 <frame-shift> <srcdir> <destdir>"
  echo "e.g.: $0 -1 data/train data/train_fs-1"
  echo "The script creates a new data directory with the features modified"
  echo "using the program shift-feats with the specified frame-shift."
  echo "This program automatically adds the prefix 'fs<frame-shift>-' to the"
  echo "utterance and speaker names. See also utils/data/shift_and_combine_feats.sh"
  exit 1
fi

frame_shift=$1
srcdir=$2
destdir=$3


if [ "$destdir" == "$srcdir" ]; then
  echo "$0: this script requires <srcdir> and <destdir> to be different."
  exit 1
fi

if [ ! -f $srcdir/feats.scp ]; then
  echo "$0: no such file $srcdir/feats.scp"
  exit 1;
fi

utt_prefix="fs$frame_shift-"
spk_prefix="fs$frame_shift-"

mkdir -p $destdir
utils/copy_data_dir.sh --utt-prefix $utt_prefix --spk-prefix $spk_prefix \
  $srcdir $destdir

if grep --quiet "'" $srcdir/feats.scp; then
  echo "$0: the input features already use single quotes. Can't proceed."
  exit 1;
fi

awk -v shift=$frame_shift 'NF == 2 {uttid=$1; feat=$2; qt="";} \
NF > 2 {idx=index($0, " "); uttid=$1; feat=substr($0, idx + 1); qt="\x27";} \
NF {print uttid " shift-feats --print-args=false --shift=" shift, qt feat qt " - |";}' \
  $destdir/feats.scp >$destdir/feats_shifted.scp
mv -f $destdir/feats_shifted.scp $destdir/feats.scp

echo "$0: Done"

