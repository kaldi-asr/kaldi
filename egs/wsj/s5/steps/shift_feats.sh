#!/usr/bin/env bash

# Copyright 2016    Vimal Manohar
# Apache 2.0

# This script is deprecated. The newer script utils/data/shift_feats.sh
# should be used instead.

# This script shifts the feats in the input data directory and creates a
# new directory <input-data>_fs<num-frames-shift> with shifted feats.
# If the shift is negative, the initial frames get truncated and the
# last frame repeated; if positive, vice versa.
# Used to prepare data for sequence training of models with
# frame_subsampling_factor != 1 (e.g. chain models).

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

if [ $# -ne 4 ]; then
   echo "This script is deprecated. The newer script utils/data/shift_feats.sh"
   echo "should be used instead."
   echo "usage: $0 [options] <frame-shift> <src-data-dir> <log-dir> <path-to-storage-dir>";
   echo "e.g.: $0 -1 data/train exp/shift-1_train mfcc"
   echo "options: "
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

num_frames_shift=$1
data_in=$2
logdir=$3
featdir=$4

utt_prefix="fs$num_frames_shift-"
spk_prefix="fs$num_frames_shift-"

# make $featdir an absolute pathname.
featdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $featdir ${PWD}`

utils/split_data.sh $data_in $nj || exit 1;

data=${data_in}_fs$num_frames_shift

mkdir -p $featdir $logdir
mkdir -p $data

utils/copy_data_dir.sh --utt-prefix $utt_prefix --spk-prefix $spk_prefix \
  $data_in $data

rm $data/feats.scp 2>/dev/null

# use "name" as part of name of the archive.
name=`basename $data`

for j in $(seq $nj); do
  # the next command does nothing unless $mfccdir/storage/ exists, see
  # utils/create_data_link.pl for more info.
  utils/create_data_link.pl $featdir/raw_feats_$name.$j.ark
done

$cmd JOB=1:$nj $logdir/shift.JOB.log \
  shift-feats --shift=$num_frames_shift \
  scp:$data_in/split$nj/JOB/feats.scp ark:- \| \
  copy-feats --compress=$compress ark:- \
  ark,scp:$featdir/raw_feats_$name.JOB.ark,$featdir/raw_feats_$name.JOB.scp || exit 1;

# concatenate the .scp files together.
for ((n=1; n<=nj; n++)); do
  cat $featdir/raw_feats_$name.$n.scp
done | awk -v nfs=$num_frames_shift '{print "fs"nfs"-"$0}'>$data/feats.scp || exit 1;

nf=`cat $data/feats.scp | wc -l`
nu=`cat $data/utt2spk | wc -l`
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully processed ($nf != $nu);"
  exit 1;
fi

echo "Succeeded shifting features for $name into $data"
