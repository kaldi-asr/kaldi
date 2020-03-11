#!/usr/bin/env bash

# Copyright    2017  Hossein Hadian
# Apache 2.0

# This script copies a data directory (which has a 'segments' file), extracting
# wav segments (according to the 'segments' file)
# so that the resulting data directory does not have a 'segments' file anymore.

nj=4
cmd=run.pl

. ./utils/parse_options.sh
. ./path.sh

if [ $# != 2 ]; then
  echo "Usage: $0 <srcdir> <destdir>"
  echo " This script copies data directory <srcdir> to <destdir> and removes"
  echo " the 'segments' file by extracting the wav segments."
  echo "Options: "
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  exit 1;
fi


export LC_ALL=C

srcdir=$1
dir=$2
logdir=$dir/log

if ! mkdir -p $dir/data; then
  echo "$0: failed to create directory $dir/data"
  exit 1
fi
mkdir -p $logdir

set -eu -o pipefail
utils/copy_data_dir.sh $srcdir $dir

split_segments=""
for n in $(seq $nj); do
  split_segments="$split_segments $logdir/segments.$n"
done

utils/split_scp.pl $srcdir/segments $split_segments

$cmd JOB=1:$nj $logdir/extract_wav_segments.JOB.log \
     extract-segments scp,p:$srcdir/wav.scp $logdir/segments.JOB \
     ark,scp:$dir/data/wav_segments.JOB.ark,$dir/data/wav_segments.JOB.scp

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $dir/data/wav_segments.$n.scp
done > $dir/data/wav_segments.scp

cat $dir/data/wav_segments.scp | awk '{ print $1 " wav-copy " $2 " - |" }' >$dir/wav.scp
rm $dir/{segments,reco2file_and_channel} 2>/dev/null || true
