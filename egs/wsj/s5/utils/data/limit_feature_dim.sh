#!/usr/bin/env bash

# Copyright 2016  Alibaba Robotics Corp. (author: Xingyu Na)
# Apache 2.0

# The script creates a new data directory by selecting a specified
# dimension range of the features in the source directory.

. utils/parse_options.sh

if [ $# != 3 ]; then
  echo "Usage: "
  echo "  $0 <feat-dim-range> <srcdir> <destdir>"
  echo "The script creates a new data directory by selecting a specified"
  echo "dimension range of the features in the source directory."
  echo "e.g.:"
  echo " $0 0:39 data/train_hires_pitch data/train_hires"
  exit 1;
fi

feat_dim_range=$1
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

mkdir -p $destdir
utils/copy_data_dir.sh $srcdir $destdir

if [ -f $destdir/cmvn.scp ]; then
  rm $destdir/cmvn.scp
  echo "$0: warning: removing $destdir/cmvn.cp, you will have to regenerate it from the features."
fi

rm $destdir/feats.scp
sed 's/$/\[:,'${feat_dim_range}'\]/' $srcdir/feats.scp | \
  utils/data/normalize_data_range.pl > $destdir/feats.scp

[ ! -f $srcdir/text ] && validate_opts="$validate_opts --no-text"
utils/validate_data_dir.sh $validate_opts $destdir
