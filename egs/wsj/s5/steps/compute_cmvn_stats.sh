#!/bin/bash 

# Copyright 2012  Daniel Povey
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Compute cepstral mean and variance statistics per speaker.  
# We do this in just one job; it's fast.
# This script takes no options.

if [ $# != 3 ]; then
   echo "usage: compute_cmvn_stats.sh [options] <data-dir> <log-dir> <path-to-cmvn-dir>";
   exit 1;
fi

if [ -f path.sh ]; then . ./path.sh; fi

data=$1
logdir=$2
cmvndir=$3

# make $cmvndir an absolute pathname.
cmvndir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $cmvndir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $cmvndir || exit 1;
mkdir -p $logdir || exit 1;


required="$data/feats.scp"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_cmvn.sh: no such file $f"
    exit 1;
  fi
done

 
! compute-cmvn-stats --spk2utt=ark:$data/spk2utt scp:$data/feats.scp ark,scp:$cmvndir/cmvn_$name.ark,$cmvndir/cmvn_$name.scp \
  2> $logdir/cmvn_$name.log && echo "Error computing CMVN stats" && exit 1;

cp $cmvndir/cmvn_$name.scp $data/cmvn.scp || exit 1;

nc=`cat $data/cmvn.scp | wc -l` 
nu=`cat $data/spk2utt | wc -l` 
if [ $nc -ne $nu ]; then
  echo "Error: it seems not all of the speakers got cmvn stats ($nf != $nu);"
  exit 1;
fi

echo "Succeeded creating CMVN stats for $name"
