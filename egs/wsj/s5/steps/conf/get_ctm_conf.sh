#!/usr/bin/env bash
# Copyright Johns Hopkins University (Author: Daniel Povey) 2012.  Apache 2.0.

# This script produces CTM files from a decoding directory that has lattices
# present.  This version gives you confidence scores using MBR decoding.
# See also steps/get_ctm.sh


# begin configuration section.
cmd=run.pl
stage=0
min_lmwt=5
max_lmwt=20
use_segments=true # if we have a segments file, use it to convert
                  # the segments to be relative to the original files.
iter=final
beam=5  # pruning beam before MBR decoding
#end configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "This script produces CTM files from a decoding directory that has lattices "
  echo "present.  This version gives you confidence scores using MBR decoding."
  echo "Usage: $0 [options] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "    --use-segments (true|false)     # use segments and reco2file_and_channel files "
  echo "                                    # to produce a ctm relative to the original audio"
  echo "                                    # files, with channel information (typically needed"
  echo "                                    # for NIST scoring)."
  echo "e.g.:"
  echo "$0 data/train data/lang exp/tri4a/decode/"
  echo "See also: steps/get_ctm.sh, steps/get_ctm_conf_fast.sh"
  exit 1;
fi

data=$1
lang=$2 # Note: may be graph directory not lang directory, but has the necessary stuff copied.
dir=$3

model=$dir/../$iter.mdl # assume model one level up from decoding dir.


for f in $lang/words.txt $model $dir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

name=`basename $data`; # e.g. eval2000

mkdir -p $dir/scoring/log

frame_shift_opt=
if [ -f $dir/../frame_shift ]; then
  frame_shift_opt="--frame-shift=$(cat $dir/../frame_shift)"
  echo "$0: $dir/../frame_shift exists, using $frame_shift_opt"
elif [ -f $dir/../frame_subsampling_factor ]; then
  factor=$(cat $dir/../frame_subsampling_factor) || exit 1
  frame_shift_opt="--frame-shift=0.0$factor"
  echo "$0: $dir/../frame_subsampling_factor exists, using $frame_shift_opt"
fi

if [ $stage -le 0 ]; then
  if [ -f $data/segments ] && $use_segments; then
    f=$data/reco2file_and_channel
    [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
    filter_cmd="utils/convert_ctm.pl $data/segments $data/reco2file_and_channel"
  else
    filter_cmd=cat
  fi

  nj=$(cat $dir/num_jobs)
  lats=$(for n in $(seq $nj); do echo -n "$dir/lat.$n.gz "; done)
  if [ -f $lang/phones/word_boundary.int ]; then
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/get_ctm.LMWT.log \
      set -o pipefail '&&' mkdir -p $dir/score_LMWT/ '&&' \
      lattice-prune --inv-acoustic-scale=LMWT --beam=$beam "ark:gunzip -c $lats|" ark:- \| \
      lattice-align-words $lang/phones/word_boundary.int $model ark:- ark:- \| \
      lattice-to-ctm-conf $frame_shift_opt --decode-mbr=true --inv-acoustic-scale=LMWT ark:- - \| \
      utils/int2sym.pl -f 5 $lang/words.txt \| \
      $filter_cmd '>' $dir/score_LMWT/$name.ctm || exit 1;
  else
    if [ ! -f $lang/phones/align_lexicon.int ]; then
      echo "$0: neither $lang/phones/word_boundary.int nor $lang/phones/align_lexicon.int exists: cannot align."
      exit 1;
    fi
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/get_ctm.LMWT.log \
      set -o pipefail '&&' mkdir -p $dir/score_LMWT/ '&&' \
      lattice-prune --inv-acoustic-scale=LMWT --beam=$beam "ark:gunzip -c $lats|" ark:- \| \
      lattice-align-words-lexicon $lang/phones/align_lexicon.int $model ark:- ark:- \| \
      lattice-to-ctm-conf $frame_shift_opt --decode-mbr=true --inv-acoustic-scale=LMWT ark:- - \| \
      utils/int2sym.pl -f 5 $lang/words.txt \| \
      $filter_cmd '>' $dir/score_LMWT/$name.ctm || exit 1;
  fi
fi

