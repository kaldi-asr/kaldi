#!/bin/bash

# Copyright 2017  Johns Hopkins University (author: Hossein Hadian)
# Apache 2.0

# This script copies the necessary parts of an online ivector directory
# optionally applying a mapping to the ivector_online.scp file

utt2orig=

. utils/parse_options.sh

if [ $# != 2 ]; then
  echo "Usage: "
  echo "  $0 [options] <srcdir> <destdir>"
  echo "e.g.:"
  echo " $0 exp/nnet3/online_ivector_train exp/nnet3/online_ivector_train_fs"
  echo "Options"
  echo "   --utt2orig=<file>     # utterance id mapping to use"
  exit 1;
fi


srcdir=$1
destdir=$2

if [ ! -f $srcdir/ivector_period ]; then
  echo "$0: no such file $srcdir/ivector_period"
  exit 1;
fi

if [ "$destdir" == "$srcdir" ]; then
  echo "$0: this script requires <srcdir> and <destdir> to be different."
  exit 1
fi

set -e;

mkdir -p $destdir
cp -r $srcdir/{conf,ivector_period} $destdir
if [ -z $utt2orig ]; then
  cp $srcdir/ivector_online.scp $destdir
else
  utils/apply_map.pl -f 2 $srcdir/ivector_online.scp < $utt2orig > $destdir/ivector_online.scp
fi
cp $srcdir/final.ie.id $destdir

echo "$0: Copied necessary parts of online ivector directory $srcdir to $destdir"
