#!/bin/bash

# Copyright 2016 Pegah Ghahremani
# Apache 2.0

# Compute random offsets per speaker w.r.t speakers's covariance matrix.
# It first computes mean per speaker and global mean.

# Begin configuration section.
cmd=run.pl
num_cmn_offsets=5               # Number of cmn offset used to generate random offsets.
preserve_total_covariance=false # If true, the total covariance for random offset of spks are preserved.
cmn_offset_scale=1.0            # offset scale used to scale the covariance matrix.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ];then 
  echo "Usage: $0 [opts] <data-dir> <log-dir> <offset-dir>"
  echo " e.g.: $0 data/train exp/make_offset/train offsets"
  exit 1;
fi

data=$1
logdir=$2
offsetdir=$3


# make $offsetdir an absolute pathname.
offsetdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $offsetdir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $offsetdir || exit 1;
mkdir -p $logdir || exit 1;


required_files="$data/feats.scp $data/spk2utt"

for f in $required_files; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f"
    exit 1;
  fi
done

$cmd $logdir/compute_rand_offset.log \ 
  generate-random-cmn-offsets --num-cmn-offsets=$num_cmn_offsets --preserve-total-covariance=$preserve_total_covariance ark:$data/spk2utt scp:$data/feats.scp ark,scp:$offsetdir/offsets.ark,$data/offsets.scp || exit 1;

echo "Succeeded creating random offsets for $name"
