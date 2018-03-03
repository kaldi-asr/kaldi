#!/bin/bash

# Copyright    2017  Hossein Hadian
# Apache 2.0

# This script copies a data directory (which has a 'segments' file), extracting
# wav segments (according to the 'segments' file)
# so that the resulting data directory does not have a 'segments' file anymore.

. utils/parse_options.sh
. ./path.sh

if [ $# != 2 ]; then
  echo "Usage: $0 <srcdir> <destdir>"
  echo " This script copies data directory <srcdir> to <destdir> and gets"
  echo "rid of the 'segments' file by extracting the wav segments."
  exit 1;
fi


export LC_ALL=C

srcdir=$1
dir=$2


if ! mkdir -p $dir/data; then
  echo "$0: failed to create directory $dir/data"
  exit 1
fi

set -e -o pipefail
utils/copy_data_dir.sh $srcdir $dir

extract-segments scp:$srcdir/wav.scp $srcdir/segments \
                 ark,scp:$dir/data/wav_segments.ark,$dir/data/wav_segments.scp
cat $dir/data/wav_segments.scp | awk '{ print $1 " wav-copy " $2 " - |" }' >$dir/wav.scp
rm $dir/reco2file_and_channel || true
