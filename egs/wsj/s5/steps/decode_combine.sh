#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# Combine two decoding directories by composing the lattices (we
# apply a weight to each of the original weights, by default 0.5 each).
# Note, this is not the only combination method, or the most normal combination
# method.  See also egs/wsj/s5/local/score_combine.sh.

# Begin configuration section.
weight1=0.5 # Weight on 1st set of lattices.
cmd=run.pl
skip_scoring=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 5 ]; then
  echo "Usage: steps/decode_combine.sh [options] <data> <lang-dir|graph-dir> <decode-dir1> <decode-dir2> <decode-dir-out>"
  echo " e.g.: steps/decode_combine.sh data/lang data/test exp/dir1/decode exp/dir2/decode exp/combine_1_2/decode"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                   # config containing options"
  echo "  --cmd <cmd>                              # Command to run in parallel with"
  echo "  --weight1 <weight>                       # Weight on 1st set of lattices (default 0.5)"
  exit 1;
fi

data=$1
lang_or_graphdir=$2
srcdir1=$3
srcdir2=$4
dir=$5

for f in $data/utt2spk $lang_or_graphdir/phones.txt $srcdir1/lat.1.gz $srcdir2/lat.1.gz; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

nj1=`cat $srcdir1/num_jobs` || exit 1;
nj2=`cat $srcdir2/num_jobs` || exit 1;
[ $nj1 -ne $nj2 ] && echo "$0: mismatch in number of jobs $nj1 versus $nj2" && exit 1;
nj=$nj1

mkdir -p $dir/log
echo $nj > $dir/num_jobs

# The lattice-interp command does the score interpolation (with composition),
# and the lattice-copy-backoff replaces the result with the 1st lattice, in
# cases where the composed result was empty.
$cmd JOB=1:$nj $dir/log/interp.JOB.log \
  lattice-interp --alpha=$weight1 "ark:gunzip -c $srcdir1/lat.JOB.gz|" \
   "ark,s,cs:gunzip -c $srcdir2/lat.JOB.gz|" ark:- \| \
  lattice-copy-backoff "ark,s,cs:gunzip -c $srcdir1/lat.JOB.gz|" ark,s,cs:- \
   "ark:|gzip -c >$dir/lat.JOB.gz" || exit 1;

if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh --cmd "$cmd" $data $lang_or_graphdir $dir ||
    { echo "$0: Scoring failed. (ignore by '--skip-scoring true')"; exit 1; }
fi

exit 0;
