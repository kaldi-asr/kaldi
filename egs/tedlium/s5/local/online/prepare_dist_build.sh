#!/bin/bash

# Copyright 2015  Johns Hopkins University (Author: Vijayaditya Peddinti)
# Apache 2.0
# Script to prepare the distribution from the online-nnet build

other_files= #other files to be included in the build

echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
   echo "Usage: $0 <lang-dir> <model-dir> <output-tgz>"
   echo "e.g.: $0 data/lang exp/nnet2_online/nnet_ms_a_online tedlium.tgz"
   exit 1;
fi

lang=$1
modeldir=$2
tgzfile=$3

for f in $lang/phones.txt $other_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

conf_files="ivector_extractor.conf mfcc.conf online_cmvn.conf online_nnet2_decoding.conf splice.conf"
ivec_extractor_files="final.dubm final.ie final.mat global_cmvn.stats online_cmvn.conf splice_opts"
build_files=
for d in $modeldir/conf $modeldir/ivector_extractor; do
  [ ! -d $d ] && echo "$0: no such directory $d" && exit 1;
done
for f in $ivec_extractor_files; do
  f=$modeldir/ivector_extractor/$f
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
  build_files="$build_files $f"
done

for f in $conf_files; do 
  f=$modeldir/conf/$f
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
  build_files="$build_files $f"
done

tar -czvf $tgzfile $lang $build_files $other_files  >/dev/null
