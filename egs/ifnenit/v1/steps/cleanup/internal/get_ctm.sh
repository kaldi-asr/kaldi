#!/bin/bash
# Copyright Johns Hopkins University (Author: Daniel Povey) 2012.  Apache 2.0.
# Copyright 2017  Vimal Manohar

# This script produces CTM files from a decoding directory that has lattices
# present.
# This is similar to get_ctm.sh, but gets the 
# CTM at the utterance-level.


# begin configuration section.
cmd=run.pl
stage=0
frame_shift=0.01
lmwt=10
print_silence=false
#end configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "    --frame-shift (default=0.01)    # specify this if your lattices have a frame-shift"
  echo "                                    # not equal to 0.01 seconds"
  echo "e.g.:"
  echo "$0 data/train data/lang exp/tri4a/decode/"
  echo "See also: steps/get_train_ctm.sh"
  exit 1;
fi

data=$1
lang=$2 # Note: may be graph directory not lang directory, but has the necessary stuff copied.
dir=$3

if [ -f $dir/final.mdl ]; then
  model=$dir/final.mdl
else
  model=$dir/../final.mdl # assume model one level up from decoding dir.
fi

for f in $lang/words.txt $model $dir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

name=`basename $data`; # e.g. eval2000

mkdir -p $dir/scoring/log

if [ $stage -le 0 ]; then
  nj=$(cat $dir/num_jobs)
  if [ -f $lang/phones/word_boundary.int ]; then
    $cmd JOB=1:$nj $dir/scoring/log/get_ctm.JOB.log \
      set -o pipefail '&&' mkdir -p $dir/score_$lmwt/ '&&' \
      lattice-1best --lm-scale=$lmwt "ark:gunzip -c $dir/lat.JOB.gz|" ark:- \| \
      lattice-align-words $lang/phones/word_boundary.int $model ark:- ark:- \| \
      nbest-to-ctm --frame-shift=$frame_shift --print-silence=$print_silence ark:- - \| \
      utils/int2sym.pl -f 5 $lang/words.txt \
      '>' $dir/score_$lmwt/${name}.ctm.JOB || exit 1;
  elif [ -f $lang/phones/align_lexicon.int ]; then
    $cmd JOB=1:$nj $dir/scoring/log/get_ctm.JOB.log \
      set -o pipefail '&&' mkdir -p $dir/score_$lmwt/ '&&' \
      lattice-1best --lm-scale=$lmwt "ark:gunzip -c $dir/lat.JOB.gz|" ark:- \| \
      lattice-align-words-lexicon $lang/phones/align_lexicon.int $model ark:- ark:- \| \
      lattice-1best ark:- ark:- \| \
      nbest-to-ctm --frame-shift=$frame_shift --print-silence=$print_silence ark:- - \| \
      utils/int2sym.pl -f 5 $lang/words.txt \
      '>' $dir/score_LMWT/${name}.ctm.JOB || exit 1;
  else
    echo "$0: neither $lang/phones/word_boundary.int nor $lang/phones/align_lexicon.int exists: cannot align."
    exit 1;
  fi
fi



