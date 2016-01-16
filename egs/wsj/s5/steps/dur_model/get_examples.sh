#!/bin/bash

# Copyright 2015 Hossein Hadian

cmd=run.pl

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
   echo "Usage: $0 [options] <ali-dir> <duration-model-dir>"
   echo "e.g.: $0 exp/mono_ali exp/mono/durmod"
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
fi

numparts=20
all_parts=$(for i in $(seq -s ' ' $numparts); do echo -n "ark,t:$dir/$i.egs.tmp "; done)
$cmd $dir/log/divide_egs.log \
     nnet3-shuffle-egs ark:$dir/all.egs ark:- \| nnet3-copy-egs ark:- $all_parts

train_parts=$(for i in $(seq -s ' ' 4 $numparts); do echo -n "$dir/$i.egs.tmp "; done)
validation_parts="$dir/1.egs.tmp $dir/2.egs.tmp $dir/3.egs.tmp"

cat $train_parts >$dir/train.egs
cat $validation_parts >$dir/val.egs


rm $dir/*.tmp
echo "$0: Done"
