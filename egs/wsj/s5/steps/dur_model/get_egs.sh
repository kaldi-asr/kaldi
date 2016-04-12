#!/bin/bash

# Copyright 2015 Hossein Hadian
# Apache 2.0.
#
# This script, which will generally be called from other nnet-duration-model
# training scripts, extracts the training examples used to train the
# nnet-duration model (and also the validation examples used for diagnostics),
# and puts them in separate archives (i.e. train.egs and val.egs)
#
# The examples are extracted from the alignment files (i.e. ali.*.gz) in an
# alignment dir (for eg. exp/tri3_ali). Each example pertains to a single phone
# (the features of which are extracted from its context:
# the phones before and after it)
# Later at the training stage, the examples are merged into minibatches.


# Begin configuration section.
cmd=run.pl
validation_ratio=5          # ratio of validation data over all data. Default is 5 (i.e. 5 percent)
shuffle_buffer_size=5000    # buffer size for nnet3-shuffle-egs to shuffle examples before 
                            # creating the train and validation sets. 

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

echo "$0 $@"  # Print the command line for logging

if [ $# != 2 ]; then
   echo "Usage: $0 [options] <ali-dir> <duration-model-dir>"
   echo "e.g.: $0 exp/mono_ali exp/mono/durmod"
   echo ""
   echo "Main options (for others, see top of script file):"
   echo "  --validation-ratio      <number>                 # ratio of validation data over all data. Default is 5 (i.e. 5 percent)"
   exit 1;
fi

alidir=$1
dir=$2

nnetdurmodel=$dir/0.mdl
transmodel=$alidir/final.mdl

for f in $nnetdurmodel $transmodel $alidir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: Required file not found: $f" && exit 1;
done

numparts=$[100/$validation_ratio]
all_parts=$(for i in $(seq -s ' ' $numparts); do echo -n "ark,t:$dir/$i.egs.tmp "; done)

$cmd $dir/log/durmod_make_examples.log \
     nnet3-durmodel-make-egs $nnetdurmodel $transmodel "ark:gunzip -c $alidir/ali.*.gz|" ark:- \
     \| nnet3-shuffle-egs --buffer-size=$shuffle_buffer_size ark:- ark:- \| nnet3-copy-egs ark:- $all_parts || exit 1;

grep -o "Wrote.*" $dir/log/durmod_make_examples.log | awk '{ print $2 }' > $dir/num_examples || exit 1;

train_parts=$(for i in $(seq -s ' ' 2 $numparts); do echo -n "$dir/$i.egs.tmp "; done)
validation_parts=$dir/1.egs.tmp
cat $train_parts >$dir/train.egs || exit 1;
cat $validation_parts >$dir/val.egs || exit 1;
rm $dir/*.tmp

echo "$0: Done"
