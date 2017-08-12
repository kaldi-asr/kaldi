#!/bin/bash

# Copyright 2016  Johns Hopkins University (author: Daniel Povey)

# Apache 2.0

# This script does the standard 3-way speed perturbing of
# a data directory (it operates on the wav.scp).

. utils/parse_options.sh

if [ $# != 2 ]; then
  echo "Usage: perturb_data_dir_speed_3way.sh <srcdir> <destdir>"
  echo "Applies standard 3-way speed perturbation using factors of 0.9, 1.0 and 1.1."
  echo "e.g.:"
  echo " $0 data/train data/train_sp"
  echo "Note: if <destdir>/feats.scp already exists, this will refuse to run."
  exit 1
fi

srcdir=$1
destdir=$2

if [ ! -f $srcdir/wav.scp ]; then
  echo "$0: expected $srcdir/wav.scp to exist"
  exit 1
fi

if [ -f $destdir/feats.scp ]; then
  echo "$0: $destdir/feats.scp already exists: refusing to run this (please delete $destdir/feats.scp if you want this to run)"
  exit 1
fi

echo "$0: making sure the utt2dur file is present in ${srcdir}, because "
echo "... obtaining it after speed-perturbing would be very slow, and"
echo "... you might need it."
utils/data/get_utt2dur.sh ${srcdir}

utils/data/perturb_data_dir_speed.sh 0.9 ${srcdir} ${destdir}_speed0.9 || exit 1
utils/data/perturb_data_dir_speed.sh 1.1 ${srcdir} ${destdir}_speed1.1 || exit 1
utils/data/combine_data.sh $destdir ${srcdir} ${destdir}_speed0.9 ${destdir}_speed1.1 || exit 1

rm -r ${destdir}_speed0.9 ${destdir}_speed1.1

echo "$0: generated 3-way speed-perturbed version of data in $srcdir, in $destdir"
utils/validate_data_dir.sh --no-feats --no-text $destdir
