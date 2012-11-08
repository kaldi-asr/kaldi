#!/bin/bash
# Copyright Johns Hopkins University (Author: Daniel Povey) 2012.  Apache 2.0.

# This script produces CTM files from a training directory that has alignments
# present.


# begin configuration section.
cmd=run.pl
stage=0
use_segments=true # if we have a segments file, use it to convert
                  # the segments to be relative to the original files.
#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: local/get_train_ctm.sh [options] <data-dir> <lang-dir> <ali-dir|exp-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "    --use-segments (true|false)     # use segments and reco2file_and_channel files "
  echo "                                    # to produce a ctm relative to the original audio"
  echo "                                    # files, with channel information (typically needed"
  echo "                                    # for NIST scoring)."
  echo "e.g.:"
  echo "local/get_train_ctm.sh data/train data/lang exp/tri3a_ali"
  echo "Produces ctm in: exp/tri3a_ali/ctm"
  exit 1;
fi

data=$1
lang=$2 # Note: may be graph directory not lang directory, but has the necessary stuff copied.
dir=$3

model=$dir/final.mdl # assume model one level up from decoding dir.


for f in $lang/words.txt $lang/phones/word_boundary.int \
     $model $dir/ali.1.gz $lang/oov.int; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

oov=`cat $lang/oov.int` || exit 1;

mkdir -p $dir/scoring/log

if [ $stage -le 0 ]; then
  if [ -f $data/segments ]; then
    f=$data/reco2file_and_channel
    [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
    filter_cmd="utils/convert_ctm.pl $data/segments $data/reco2file_and_channel"
  else
    filter_cmd=cat    
  fi

  $cmd $dir/log/get_ctm.log \
    linear-to-nbest "ark:gunzip -c $dir/ali.*.gz|" \
     "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $data/text |" \
     '' '' ark:- \| \
    lattice-align-words $lang/phones/word_boundary.int $model ark:- ark:- \| \
    nbest-to-ctm ark:- - \| \
    utils/int2sym.pl -f 5 $lang/words.txt \| \
    $filter_cmd '>' $dir/ctm || exit 1;
fi
