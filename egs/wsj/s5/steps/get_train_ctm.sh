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

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <data-dir> <lang-dir> <ali-dir|model-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "    --use-segments (true|false)     # use segments and reco2file_and_channel files "
  echo "                                    # to produce a ctm relative to the original audio"
  echo "                                    # files, with channel information (typically needed"
  echo "                                    # for NIST scoring)."
  echo "e.g.:"
  echo "$0 data/train data/lang exp/tri3a_ali"
  echo "Produces ctm in: exp/tri3a_ali/ctm"
  exit 1;
fi

data=$1
lang=$2 # Note: may be graph directory not lang directory, but has the necessary stuff copied.
dir=$3


model=$dir/final.mdl # assume model one level up from decoding dir.


for f in $lang/words.txt $model $dir/ali.1.gz $lang/oov.int; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

oov=`cat $lang/oov.int` || exit 1;
nj=`cat $dir/num_jobs` || exit 1;
split_data.sh $data $nj || exit 1;
sdata=$data/split$nj

mkdir -p $dir/log

if [ $stage -le 0 ]; then
  if [ -f $lang/phones/word_boundary.int ]; then
    $cmd JOB=1:$nj $dir/log/get_ctm.JOB.log \
      linear-to-nbest "ark:gunzip -c $dir/ali.JOB.gz|" \
      "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $sdata/JOB/text |" \
      '' '' ark:- \| \
      lattice-align-words $lang/phones/word_boundary.int $model ark:- ark:- \| \
      nbest-to-ctm ark:- - \| \
      utils/int2sym.pl -f 5 $lang/words.txt \| \
      gzip -c '>' $dir/ctm.JOB.gz
  else
    if [ ! -f $lang/phones/align_lexicon.int ]; then
      echo "$0: neither $lang/phones/word_boundary.int nor $lang/phones/align_lexicon.int exists: cannot align."
      exit 1;
    fi
    $cmd JOB=1:$nj $dir/log/get_ctm.JOB.log \
      linear-to-nbest "ark:gunzip -c $dir/ali.JOB.gz|" \
      "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $sdata/JOB/text |" \
      '' '' ark:- \| \
      lattice-align-words-lexicon $lang/phones/align_lexicon.int $model ark:- ark:- \| \
      nbest-to-ctm ark:- - \| \
      utils/int2sym.pl -f 5 $lang/words.txt \| \
      gzip -c '>' $dir/ctm.JOB.gz
  fi
fi

if [ $stage -le 1 ]; then
  if [ -f $data/segments ]; then
    f=$data/reco2file_and_channel
    [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
    for n in `seq $nj`; do gunzip -c $dir/ctm.$n.gz; done | \
      utils/convert_ctm.pl $data/segments $data/reco2file_and_channel > $dir/ctm || exit 1;
  else
    for n in `seq $nj`; do gunzip -c $dir/ctm.$n.gz; done > $dir/ctm || exit 1;
  fi
  rm $dir/ctm.*.gz
fi

