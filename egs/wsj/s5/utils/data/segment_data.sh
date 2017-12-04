#!/bin/bash

# Copyright 2013  Johns Hopkins University (author: Hossein Hadian)
# Apache 2.0


. utils/parse_options.sh
. ./path.sh

if [ $# != 2 ]; then
  echo "Usage: "
  echo "  $0 [options] <srcdir> <destdir>"
  exit 1;
fi


export LC_ALL=C

srcdir=$1
dir=$2


if ! mkdir -p $dir/data; then
  echo "$0: failed to create directory $dir"
fi

utils/copy_data_dir.sh $srcdir $dir
set -e
set -o pipefail

extract-segments scp:$srcdir/wav.scp $srcdir/segments ark,scp:$dir/data/segments.ark,$dir/segments.scp
cat $dir/segments.scp | awk '{ print $1 " wav-copy " $2 " - |" }' >$dir/wav.scp
rm $dir/reco2file_and_channel || true
