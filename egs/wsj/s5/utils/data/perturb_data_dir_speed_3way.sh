#!/bin/bash

# Copyright 2016-2018  Johns Hopkins University (author: Daniel Povey)
#                2018  Hossein Hadian

# Apache 2.0

# This script does the standard 3-way speed perturbing of
# a data directory (it operates on the wav.scp).

# If you add the option "--always-include-prefix true", it will include the
# prefix "sp1.0-" for the original un-perturbed data.  This can help resolve
# problems with sorting.
# We don't make '--always-include-prefix true' the default  behavior because
# it can break some older scripts that relied on the original utterance-ids
# being a subset of the perturbed data's utterance-ids.

always_include_prefix=false

. utils/parse_options.sh

if [ $# != 2 ]; then
  echo "Usage: perturb_data_dir_speed_3way.sh <srcdir> <destdir>"
  echo "Applies standard 3-way speed perturbation using factors of 0.9, 1.0 and 1.1."
  echo "e.g.:"
  echo " $0 [options] data/train data/train_sp"
  echo "Note: if <destdir>/feats.scp already exists, this will refuse to run."
  echo "Options:"
  echo "    --always-include-prefix [true|false]   # default: false.  If set to true,"
  echo "                                           # it will add the prefix 'sp1.0-' to"
  echo "                                           # utterance and speaker-ids for data at"
  echo "                                           # the original speed.  Can resolve"
  echo "                                           # issues RE data sorting."
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

if $always_include_prefix; then
  utils/copy_data_dir.sh --spk-prefix sp1.0- --utt-prefix sp1.0- ${srcdir} ${destdir}_speed1.0
  if [ ! -f $srcdir/utt2uniq ]; then
    cat $srcdir/utt2spk | awk  '{printf("sp1.0-%s %s\n", $1, $1);}' > ${destdir}_speed1.0/utt2uniq
  else
    cat $srcdir/utt2uniq | awk '{printf("sp1.0-%s %s\n", $1, $2);}' > ${destdir}_speed1.0/utt2uniq
  fi
  utils/data/combine_data.sh $destdir ${destdir}_speed1.0 ${destdir}_speed0.9 ${destdir}_speed1.1 || exit 1

  rm -r ${destdir}_speed0.9 ${destdir}_speed1.1 ${destdir}_speed1.0
else
  utils/data/combine_data.sh $destdir ${srcdir} ${destdir}_speed0.9 ${destdir}_speed1.1 || exit 1
  rm -r ${destdir}_speed0.9 ${destdir}_speed1.1
fi

echo "$0: generated 3-way speed-perturbed version of data in $srcdir, in $destdir"
if ! utils/validate_data_dir.sh --no-feats --no-text $destdir; then
  echo "$0: Validation failed.  If it is a sorting issue, try the option '--always-include-prefix true'."
  exit 1
fi

exit 0
