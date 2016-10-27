#!/bin/bash

# Copyright 2016 Pegah Ghahremani
# Apache 2.0

# Computes random offsets per speaker w.r.t speakers's covariance matrix.
# It first computes mean per speaker and global mean.

# Begin configuration section.
cmd=run.pl
offsets_config=conf/offsets.conf
compress=true
# End configuration section. 

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


required_files="$data/feats.scp $data/spk2utt $offsets_config"

for f in $required_files; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f"
    exit 1;
  fi
done

$cmd $logdir/compute_rand_offset.log \ 
  generate-random-cmn-offsets --config=$offsets_config ark:$data/spk2utt scp:$data/feats.scp ark:- | copy-feats --compress=$compress ark:- ark,scp:$offsetdir/offsets.ark,$data/offsets.scp || exit 1;

echo "Succeeded creating random offsets for $name"
