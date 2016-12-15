#!/bin/bash

# Copyright 2016  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

# This script takes as input a data directory, such as data/train/, preferably
# with utt2dur file already existing (or the utt2dur file will be created if
# not), and it attempts to work out the approximate frame shift by comparing the
# utt2dur with the output of feat-to-len on the feats.scp.  It prints it out.
# if the shift is very close to, but above, 0.01 (the normal frame shift) it
# rounds it down.

. utils/parse_options.sh
. ./path.sh

if [ $# != 1 ]; then
  echo "Usage: $0 <datadir>"
  echo "e.g.:"
  echo " $0 data/train"
  echo "This script prints the frame-shift (e.g. 0.01) to the standard out."
  echo "If <datadir> does not contain utt2dur, this script may call utils/data/get_utt2dur.sh,"
  echo "which will require write permission to <datadir>"
  exit 1
fi

export LC_ALL=C

dir=$1


if [ ! -s $dir/utt2dur ]; then
  if [ ! -e $dir/wav.scp ] && [ ! -s $dir/segments ]; then
    echo "$0: neither $dir/wav.scp nor $dir/segments exist; assuming a frame shift of 0.01." 1>&2
    echo 0.01
    exit 0
  fi
  echo "$0: $dir/utt2dur does not exist: creating it" 1>&2
  utils/data/get_utt2dur.sh $dir 1>&2
fi

if [ ! -f $dir/feats.scp ]; then
  echo "$0: $dir/feats.scp does not exist" 1>&2
  exit 1
fi

temp=$(mktemp /tmp/tmp.XXXX)

feat-to-len "scp:head -n 10 $dir/feats.scp|" ark,t:- > $temp

if [ -z $temp ]; then
  echo "$0: error running feat-to-len" 1>&2
  exit 1
fi

head -n 10 $dir/utt2dur | paste - $temp | \
   awk '{ dur += $2; frames += $4; } END { shift = dur / frames; if (shift > 0.01 && shift < 0.0102) shift = 0.01; print shift; }' || exit 1;

rm $temp

exit 0
