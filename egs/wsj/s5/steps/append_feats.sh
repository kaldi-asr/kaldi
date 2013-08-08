#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0
# This script appends the features in two data directories.

# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
cmd=run.pl
nj=4
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "usage: append_feats.sh [options] <src-data-dir1> <src-data-dir2> <dest-data-dir> <log-dir> <path-to-storage-dir>";
   echo "options: "
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data_src1=$1
data_src2=$2
data=$3
logdir=$4
mfccdir=$5

# make $mfccdir an absolute pathname.
mfccdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $mfccdir ${PWD}`

utils/split_data.sh $data_src1 $nj || exit 1;
utils/split_data.sh $data_src2 $nj || exit 1;

mkdir -p $mfccdir $logdir

mkdir -p $data 
cp $data_src1/* $data/ 2>/dev/null # so we get the other files, such as utt2spk.
rm $data/cmvn.scp 2>/dev/null 
rm -r $data/split* 2>/dev/null

# use "name" as part of name of the archive.
name=`basename $data`

$cmd JOB=1:$nj $logdir/append.JOB.log \
   append-feats --truncate-frames=true \
   scp:$data_src1/split$nj/JOB/feats.scp scp:$data_src2/split$nj/JOB/feats.scp \
   ark,scp:$mfccdir/appended_$name.JOB.ark,$mfccdir/appended_$name.JOB.scp || exit 1;
              
# concatenate the .scp files together.
for ((n=1; n<=nj; n++)); do
  cat $mfccdir/appended_$name.$n.scp >> $data/feats.scp || exit 1;
done > $data/feats.scp || exit 1;


nf=`cat $data/feats.scp | wc -l` 
nu=`cat $data/utt2spk | wc -l` 
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $data"
fi

echo "Succeeded creating MFCC features for $name"
