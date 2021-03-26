#!/usr/bin/env bash

# Copyright 2014  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# This script is deprecated. Use utils/data/limit_feature_dim.sh.

# This script selects some specified dimensions of the features in the
# input data directory.

# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
cmd=run.pl
nj=4
compress=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 3 ] || [ $# -gt 5 ]; then
   echo "usage: $0 [options] <selector> <src-data-dir>  <dest-data-dir> [<log-dir> [<path-to-storage-dir>] ]";
   echo "e.g.: $0 0-12 data/train_mfcc_pitch data/train_mfcconly exp/select_pitch_train mfcc"
   echo "Note: <log-dir> defaults to <data-dir>/log, and <path-to-storage-dir> defaults to <data-dir>/data"
   echo "options: "
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

selector="$1"
data_in=$2
data=$3
if [ $# -gt 3 ];then
  logdir=$4
else
  logdir=$data/log
fi

if [ $# -gt 4 ];then
  ark_dir=$5
else
  ark_dir=$data/data
fi

# make $ark_dir an absolute pathname.
ark_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $ark_dir ${PWD}`


utils/split_data.sh $data_in $nj || exit 1;

mkdir -p $ark_dir $logdir
mkdir -p $data

cp $data_in/* $data/ 2>/dev/null # so we get the other files, such as utt2spk.
rm $data/cmvn.scp 2>/dev/null
rm $data/feats.scp 2>/dev/null

# use "name" as part of name of the archive.
name=`basename $data`

for j in $(seq $nj); do
  # the next command does nothing unless $mfccdir/storage/ exists, see
  # utils/create_data_link.pl for more info.
  utils/create_data_link.pl $ark_dir/selected_$name.$j.ark
done

$cmd JOB=1:$nj $logdir/append.JOB.log \
   select-feats "$selector" scp:$data_in/split$nj/JOB/feats.scp ark:- \| \
   copy-feats --compress=$compress ark:- \
    ark,scp:$ark_dir/selected_$name.JOB.ark,$ark_dir/selected_$name.JOB.scp || exit 1;

# concatenate the .scp files together.
for ((n=1; n<=nj; n++)); do
  cat $ark_dir/selected_$name.$n.scp >> $data/feats.scp || exit 1;
done > $data/feats.scp || exit 1;


nf=`cat $data/feats.scp | wc -l`
nu=`cat $data/utt2spk | wc -l`
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully processed ($nf != $nu);"
  exit 1;
fi

echo "Succeeded selecting features for $name into $data"
