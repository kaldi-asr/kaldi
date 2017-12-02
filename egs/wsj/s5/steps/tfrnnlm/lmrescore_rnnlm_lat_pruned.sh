#!/bin/bash

# Copyright 2015  Guoguo Chen
#           2017  Hainan Xu
# Apache 2.0

# This script rescores lattices with RNNLM.  See also rnnlmrescore.sh which is
# an older script using n-best lists.

# Begin configuration section.
cmd=run.pl
skip_scoring=false
max_ngram_order=4 # Approximate the lattice-rescoring by limiting the max-ngram-order
                  # if it's set, it merges histories in the lattice if they share
                  # the same ngram history and this prevents the lattice from 
                  # exploding exponentially. Details of the n-gram approximation
                  # method are described in section 2.3 of the paper
                  # http://www.cs.jhu.edu/~hxu/tf.pdf
inv_acwt=10
weight=0.5  # Interpolation weight for RNNLM.
# End configuration section.

echo "$0 $@"  # Print the command line for logging

. ./utils/parse_options.sh

if [ $# != 5 ]; then
   echo "Does language model rescoring of lattices (remove old LM, add new LM)"
   echo "with RNNLM."
   echo ""
   echo "Usage: $0 [options] <old-lang-dir> <rnnlm-dir> \\"
   echo "                   <data-dir> <input-decode-dir> <output-decode-dir>"
   echo " e.g.: $0 ./rnnlm data/lang_tg data/test \\"
   echo "                   exp/tri3/test_tg exp/tri3/test_rnnlm"
   echo "options: [--cmd (run.pl|queue.pl [queue opts])]"
   exit 1;
fi

[ -f path.sh ] && . ./path.sh;

oldlang=$1
rnnlm_dir=$2
data=$3
indir=$4
outdir=$5

oldlm=$oldlang/G.fst

[ ! -f $oldlm ] && echo "$0: Missing file $oldlm" && exit 1;
[ ! -f $rnnlm_dir/rnnlm ] && [ ! -d $rnnlm_dir/rnnlm ] && echo "$0: Missing file $rnnlm_dir/rnnlm" && exit 1;
[ ! -f $rnnlm_dir/unk.probs ] &&\
  echo "$0: Missing file $rnnlm_dir/unk.probs" && exit 1;
[ ! -f $oldlang/words.txt ] &&\
  echo "$0: Missing file $oldlang/words.txt" && exit 1;
! ls $indir/lat.*.gz >/dev/null &&\
  echo "$0: No lattices input directory $indir" && exit 1;
awk -v n=$0 -v w=$weight 'BEGIN {if (w < 0 || w > 1) {
  print n": Interpolation weight should be in the range of [0, 1]"; exit 1;}}' \
  || exit 1;

acwt=`perl -e "print (1.0/$inv_acwt);"`

mkdir -p $outdir/log
nj=`cat $indir/num_jobs` || exit 1;
cp $indir/num_jobs $outdir

$cmd JOB=1:$nj $outdir/log/rescorelm.JOB.log \
  lattice-lmrescore-tf-rnnlm-pruned --lm-scale=$weight \
  --acoustic-scale=$acwt --max-ngram-order=$max_ngram_order \
  $oldlm $oldlang/words.txt \
  $rnnlm_dir/unk.probs $rnnlm_dir/wordlist.rnn.final "$rnnlm_dir/rnnlm" \
  "ark:gunzip -c $indir/lat.JOB.gz|" "ark,t:|gzip -c>$outdir/lat.JOB.gz" || exit 1;

if ! $skip_scoring ; then
  err_msg="Not scoring because local/score.sh does not exist or not executable."
  [ ! -x local/score.sh ] && echo $err_msg && exit 1;
  local/score.sh --cmd "$cmd" $data $oldlang $outdir
else
  echo "Not scoring because requested so..."
fi

exit 0;
