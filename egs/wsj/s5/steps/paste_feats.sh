#!/bin/bash

# Copyright 2014  Brno University of Technology (Author: Karel Vesely)
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0
# This script appends the features in two data directories.

# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
cmd=run.pl
nj=4
length_tolerance=10 # length tolerance in frames (trim to shortest)
compress=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 5 ]; then
   echo "usage: $0 [options] <src-data-dir1> <src-data-dir2> [<src-data-dirN>] <dest-data-dir> <log-dir> <path-to-storage-dir>";
   echo "options: "
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data_src_arr=(${@:1:$(($#-3))}) #array of source data-dirs
data=${@: -3: 1}
logdir=${@: -2: 1}
ark_dir=${@: -1: 1} #last arg.

data_src_first=${data_src_arr[0]} # get 1st src dir

# make $ark_dir an absolute pathname.
ark_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $ark_dir ${PWD}`

for data_src in ${data_src_arr[@]}; do
  utils/split_data.sh $data_src $nj || exit 1;
done

mkdir -p $ark_dir $logdir

mkdir -p $data 
cp $data_src_first/* $data/ 2>/dev/null # so we get the other files, such as utt2spk.
rm $data/cmvn.scp 2>/dev/null 
rm $data/feats.scp 2>/dev/null 

# use "name" as part of name of the archive.
name=`basename $data`

# get list of source scp's for pasting
data_src_args=
for data_src in ${data_src_arr[@]}; do
  data_src_args="$data_src_args scp:$data_src/split$nj/JOB/feats.scp"
done

$cmd JOB=1:$nj $logdir/append.JOB.log \
   paste-feats --length-tolerance=$length_tolerance $data_src_args ark:- \| \
   copy-feats --compress=$compress ark:- \
    ark,scp:$ark_dir/pasted_$name.JOB.ark,$ark_dir/pasted_$name.JOB.scp || exit 1;
              
# concatenate the .scp files together.
for ((n=1; n<=nj; n++)); do
  cat $ark_dir/pasted_$name.$n.scp >> $data/feats.scp || exit 1;
done > $data/feats.scp || exit 1;


nf=`cat $data/feats.scp | wc -l` 
nu=`cat $data/utt2spk | wc -l` 
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $data"
fi

echo "Succeeded pasting features for $name into $data"
