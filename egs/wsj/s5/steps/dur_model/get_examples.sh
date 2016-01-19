#!/bin/bash

# Copyright 2015 Hossein Hadian

cmd=run.pl
validation_ratio=5
shuffle_buffer_size=5000

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
   echo "Usage: $0 [options] <ali-dir> <duration-model-dir>"
   echo "e.g.: $0 exp/mono_ali exp/mono/durmod"
   echo ""
   echo "Main options (for others, see top of script file):"
   echo "  --validation-ratio      <number>                 # ratio of validation data over all data. default is 5 (i.e. 5 percent)"   
   exit 1;
fi

alidir=$1
dir=$2

durmodel=$dir/durmodel.mdl
transmodel=$alidir/final.mdl

for f in $transmodel $alidir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: Required file not found: $f" && exit 1;
done

if [ ! -f $dir/all.ali ]; then
  gunzip -c $alidir/ali.*.gz >> $dir/all.ali || exit 1;
fi

if [ ! -f $dir/all.egs ]; then
  $cmd $dir/log/durmod_make_examples.log \
       nnet3-durmodel-make-egs $durmodel $transmodel ark:$dir/all.ali ark,t:$dir/all.egs || exit 1;
  grep -o "Wrote.*" $dir/log/durmod_make_examples.log | awk '{ print $2 }' > $dir/num_examples
else
  echo "$0: all.egs already exists. Not overwriting..."
fi

if [[ ! -f $dir/train.egs || ! -f $dir/val.egs || $dir/train.egs -ot $dir/all.egs ]] ; then
  numparts=$[100/$validation_ratio]
  all_parts=$(for i in $(seq -s ' ' $numparts); do echo -n "ark,t:$dir/$i.egs.tmp "; done)
  $cmd $dir/log/divide_egs.log \
       nnet3-shuffle-egs --buffer-size=$shuffle_buffer_size ark:$dir/all.egs ark:- \| nnet3-copy-egs ark:- $all_parts
  train_parts=$(for i in $(seq -s ' ' 2 $numparts); do echo -n "$dir/$i.egs.tmp "; done)
  validation_parts=$dir/1.egs.tmp
  cat $train_parts >$dir/train.egs
  cat $validation_parts >$dir/val.egs
  rm $dir/*.tmp
else
  echo "$0: train.egs and val.egs already exist. Not overwriting..."
fi

echo "$0: Done"
