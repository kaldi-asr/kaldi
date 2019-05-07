#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
#           2017  Vimal Manohar
#           2018  Xiaohui Zhang
#           2018  Music Technology Group, Universitat Pompeu Fabra.
# Apache 2.0

# This script produces CTM files from a decoding directory that has lattices
# present. It does this for one LM weight and also supports 
# the word insertion penalty.
# This is similar to get_ctm.sh, but gets the CTM at the utterance-level.
# It can be faster than steps/get_ctm.sh --use-segments false as it splits
# the process across many jobs. 

# begin configuration section.
cmd=run.pl
stage=0
frame_shift=0.01
lmwt=10
wip=0.0
print_silence=false
#end configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 4 ]; then
  echo "Usage: $0 [options] <data-dir> <lang-dir|graph-dir> <decode-dir> <ctm-out-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "    --frame-shift (default=0.01)    # specify this if your lattices have a frame-shift"
  echo "                                    # not equal to 0.01 seconds"
  echo "e.g.:"
  echo "$0 data/train data/lang exp/tri4a/decode/"
  echo "See also: steps/get_ctm.sh, steps/get_ctm_conf.sh"
  exit 1;
fi

data=$1
lang=$2 # Note: may be graph directory not lang directory, but has the necessary stuff copied.
decode_dir=$3
dir=$4

if [ -f $decode_dir/final.mdl ]; then
  model=$decode_dir/final.mdl
else
  model=$decode_dir/../final.mdl # assume model one level up from decoding dir.
fi

for f in $lang/words.txt $model $decode_dir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

mkdir -p $dir

nj=$(cat $decode_dir/num_jobs)
echo $nj > $dir/num_jobs

if [ -f $lang/phones/word_boundary.int ]; then
  $cmd JOB=1:$nj $dir/log/get_ctm.JOB.log \
    set -o pipefail '&&' \
    lattice-1best --lm-scale=$lmwt --word-ins-penalty=$wip "ark:gunzip -c $decode_dir/lat.JOB.gz|" ark:- \| \
    lattice-align-words $lang/phones/word_boundary.int $model ark:- ark:- \| \
    nbest-to-ctm --frame-shift=$frame_shift --print-silence=$print_silence ark:- - \| \
    utils/int2sym.pl -f 5 $lang/words.txt \
    '>' $dir/ctm.JOB || exit 1;
elif [ -f $lang/phones/align_lexicon.int ]; then
  $cmd JOB=1:$nj $dir/log/get_ctm.JOB.log \
    set -o pipefail '&&' \
    lattice-1best --lm-scale=$lmwt --word-ins-penalty=$wip "ark:gunzip -c $decode_dir/lat.JOB.gz|" ark:- \| \
    lattice-align-words-lexicon $lang/phones/align_lexicon.int $model ark:- ark:- \| \
    lattice-1best ark:- ark:- \| \
    nbest-to-ctm --frame-shift=$frame_shift --print-silence=$print_silence ark:- - \| \
    utils/int2sym.pl -f 5 $lang/words.txt \
    '>' $dir/ctm.JOB || exit 1;
else
  echo "$0: neither $lang/phones/word_boundary.int nor $lang/phones/align_lexicon.int exists: cannot align."
  exit 1;
fi

for n in `seq $nj`; do 
  cat $dir/ctm.$n
done > $dir/ctm
