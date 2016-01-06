#!/bin/bash

# Copyright 2015 Hossein Hadian

# This script calls all the duration modeling scripts with default settings and does the
# rescoring.

testdata=
cmd=run.pl

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Usage: $0 [options] <phones-dir> <ali-dir> <decode-dir> <rescored-decode-dir>"
   echo "e.g.: $0 data/lang/phones exp/mono_ali exp/mono/decode_test_bg exp/mono/decode_test_bg_durmod"     
   exit 1;
fi

phones_dir=$1
alidir=$2
decode_dir=$3
dir=$4

steps/durmod/durmodel_prepare_examples.sh --left-context 4 \
                                     --right-context 2 \
                                     --max-duration 15 \
                                     $phones_dir $alidir $alidir/durmod

steps/durmod/durmodel_train.sh --num-epochs 100 --minibatch-size 512 $alidir/durmod

steps/durmod/durmodel_rescore.sh --lm-scale 0.75 $alidir/durmod/durmodel.mdl \
                                $decode_dir $dir
if [[ ! -z $testdata ]]; then
  lang=$(dirname $phones_dir)
  local/score.sh $testdata $lang $dir
  grep WER  $dir/wer*
fi

