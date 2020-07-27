#!/usr/bin/env bash

# Copyright 2014  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# This is as prepare_online_decoding.sh, but for transfer learning-- the case where
# you have an existing online-decoding directory where you have all the feature
# stuff, that you don't want to change, but 

# Begin configuration.
stage=0 # This allows restarting after partway, when something went wrong.
cmd=run.pl
iter=final
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# -ne 4 ]; then    
  echo "Usage: $0 [options] <orig-nnet-online-dir> <new-lang-dir> <new-nnet-dir> <new-nnet-online-dir>"
  echo "e.g.: $0 exp_other/nnet2_online/nnet_a_online data/lang exp/nnet2_online/nnet_a exp/nnet2_online/nnet_a_online"
  echo "main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  exit 1;
fi

online_src=$1
lang=$2
nnet_src=$3
dir=$4

for f in $online_src/conf/online_nnet2_decoding.conf $nnet_src/final.mdl $nnet_src/tree $lang/words.txt; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done


dir_as_given=$dir
dir=$(utils/make_absolute.sh $dir) # Convert $dir to an absolute pathname, so that the
                        # configuration files we write will contain absolute
                        # pathnames.
mkdir -p $dir/conf $dir/log

utils/lang/check_phones_compatible.sh $lang/phones.txt $nnet_src/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

cp $nnet_src/tree $dir/ || exit 1;

cp $nnet_src/$iter.mdl $dir/ || exit 1;


# There are a bunch of files that we will need to copy from $online_src, because
# we're aiming to have one self-contained directory that has everything in it.
mkdir -p $dir/ivector_extractor
cp -r $online_src/ivector_extractor/* $dir/ivector_extractor

[ ! -d $online_src/conf ] && \
  echo "Expected directory $online_src/conf to exist" && exit 1;

for x in $online_src/conf/*conf; do
  # Replace directory name starting $online_src with those starting with $dir.
  # We actually replace any directory names ending in /ivector_extractor/ or /conf/ 
  # with $dir/ivector_extractor/ or $dir/conf/
  cat $x | perl -ape "s:=(.+)/(ivector_extractor|conf)/:=$dir/\$2/:;" > $dir/conf/$(basename $x)
done


# modify the silence-phones in the config; these are only used for the
# endpointing code.
cp $dir/conf/online_nnet2_decoding.conf{,.tmp}
silphones=$(cat $lang/phones/silence.csl) || exit 1;
cat $dir/conf/online_nnet2_decoding.conf.tmp | \
  sed s/silence-phones=.\\+/silence-phones=$silphones/ > $dir/conf/online_nnet2_decoding.conf
rm $dir/conf/online_nnet2_decoding.conf.tmp

echo "$0: formatted neural net for online decoding in $dir_as_given"
