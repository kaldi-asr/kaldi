#!/bin/bash

# Copyright 2014  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# This is as prepare_online_decoding.sh, but it's for a special case, where we
# already have a directory that's been prepared in that way, but for another
# corpus, and we have used the script
# steps/online/nnet2/dump_nnet_activations.sh to dump activations of the last
# hidden layer of that network on our data, and then steps/nnet2/retrain_fast.sh
# to train a neural net on top of those activations.  The job of this script is
# to take the original neural net, and the net that was trained on top of
# its last hidden layer, combine them, and create an online-decoding directory
# in the same format as is created by prepare_online_decoding.sh.
# All the options for the feature extraction and the iVector extractor
# are taken from the original directory from the other corpus.


# Begin configuration.
stage=0 # This allows restarting after partway, when something when wrong.
cleanup=true
cmd=run.pl
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# -ne 3 ] && [ $# -ne 4 ]; then    
  echo "Usage: $0 [options] <orig-nnet-online-dir> [<new-lang-dir>] <new-nnet-dir> <new-nnet-online-dir>"
  echo "e.g.: $0 exp_other/nnet2_online/nnet_a_online data/lang exp/nnet2_online/nnet_a exp/nnet2_online/nnet_a_online"
  echo "main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  exit 1;
fi

if [ $# -eq 3 ]; then
  echo "$0: warning: it's better if you add the new <lang> directory as the 2nd argument."

  online_src=$1
  lang=
  nnet_src=$2
  dir=$3
else
  online_src=$1
  lang=$2
  nnet_src=$3
  dir=$4

  extra_files=$lang/words.txt
fi


for f in $online_src/conf/online_nnet2_decoding.conf $nnet_src/final.mdl $nnet_src/tree $extra_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done


dir_as_given=$dir
dir=$(readlink -f $dir) # Convert $dir to an absolute pathname, so that the
                        # configuration files we write will contain absolute
                        # pathnames.
mkdir -p $dir/conf $dir/log


cp $nnet_src/tree $dir/ || exit 1;


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

info=$dir/nnet_info
nnet-am-info $online_src/final.mdl >$info
nc=$(grep num-components $info | awk '{print $2}');
if grep SumGroupComponent $info >/dev/null; then 
  nc_truncate=$[$nc-3]  # we did mix-up: remove AffineComponent,
                          # SumGroupComponent, SoftmaxComponent
else
  nc_truncate=$[$nc-2]  # remove AffineComponent, SoftmaxComponent
fi
$cmd $dir/log/get_raw_nnet.log \
 nnet-to-raw-nnet --truncate=$nc_truncate $online_src/final.mdl $dir/first_nnet.raw || exit 1;

# Now create the final.mdl, by inserting $dir/first_nnet.raw at the beginning
# of the model in $nnet_src/final.mdl

$cmd $dir/log/append_nnet.log \
  nnet-insert --randomize-next-component=false --insert-at=0 \
  $nnet_src/final.mdl $dir/first_nnet.raw $dir/final.mdl || exit 1;

$cleanup && rm $dir/first_nnet.raw

if [ ! -z "$lang" ]; then
  # if the $lang option was provided, modify the silence-phones in the config;
  # these are only used for the endpointing code, but we should get this right.
  cp $dir/conf/online_nnet2_decoding.conf{,.tmp}
  silphones=$(cat $lang/phones/silence.csl) || exit 1;
  cat $dir/conf/online_nnet2_decoding.conf.tmp | \
    sed s/silence-phones=.\\+/silence-phones=$silphones/ > $dir/conf/online_nnet2_decoding.conf
  rm $dir/conf/online_nnet2_decoding.conf.tmp
fi

echo "$0: formatted neural net for online decoding in $dir_as_given"
