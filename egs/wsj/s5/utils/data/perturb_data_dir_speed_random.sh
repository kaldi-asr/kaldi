#!/bin/bash

# Copyright 2017  Vimal Manohar

# Apache 2.0

speeds="0.9 1.0 1.1"

. utils/parse_options.sh

if [ $# != 2 ]; then
  echo "Usage: perturb_data_dir_speed_random.sh <srcdir> <destdir>"
  echo "Applies 3-way speed perturbation using factors of 0.9, 1.0 and 1.1 on random subsets."
  echo "e.g.:"
  echo " $0 data/train data/train_spr"
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

num_speeds=`echo $speeds | awk '{print NF}'`
utils/split_data.sh --per-reco $srcdir $num_speeds

speed_dirs=
i=1
for speed in $speeds; do 
  if [ $speed != 1.0 ]; then
    utils/data/perturb_data_dir_speed.sh $speed ${srcdir}/split${num_speeds}reco/$i ${destdir}_speed$speed || exit 1
    speed_dirs="${speed_dirs} ${destdir}_speed$speed"
  else 
    speed_dirs="$speed_dirs ${srcdir}/split${num_speeds}reco/$i"
  fi
done

utils/data/combine_data.sh $destdir ${speed_dirs} || exit 1

rm -r $speed_dirs ${srcdir}/split${num_speeds}reco

echo "$0: generated $num_speeds-way speed-perturbed version of random subsets of data in $srcdir, in $destdir"
utils/validate_data_dir.sh --no-feats --no-text $destdir
