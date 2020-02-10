#!/usr/bin/env bash

# Copyright 2015  Guoguo Chen
#           2017  Hainan Xu
# Apache 2.0

# This script rescores lattices with RNNLM trained with TensorFlow.
# A faster and more accurate version of the algorithm is at
# steps/tfrnnlm/lmrescore_rnnlm_lat_pruned.sh which is prefered
# One example recipe of this script is at egs/ami/s5/local/tfrnnlm/run_lstm_fast.sh

# Begin configuration section.
cmd=run.pl
skip_scoring=false
max_ngram_order=4 # Approximate the lattice-rescoring by limiting the max-ngram-order
                  # if it's set, it merges histories in the lattice if they share
                  # the same ngram history and this prevents the lattice from 
                  # exploding exponentially. Details of the n-gram approximation
                  # method are described in section 2.3 of the paper
                  # http://www.danielpovey.com/files/2018_icassp_lattice_pruning.pdf
weight=0.5  # Interpolation weight for RNNLM.
# End configuration section.

echo "$0 $@"  # Print the command line for logging

. ./utils/parse_options.sh

if [ $# != 5 ]; then
   echo "Does language model rescoring of lattices (remove old LM, add new LM)"
   echo "with TensorFlow RNNLM."
   echo ""
   echo "Usage: $0 [options] <old-lang-dir> <rnnlm-dir> \\"
   echo "                   <data-dir> <input-decode-dir> <output-decode-dir>"
   echo " e.g.: $0 data/lang_tg data/tensorflow_lstm data/test \\"
   echo "                   exp/tri3/test_tg exp/tri3/test_tfrnnlm"
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
if [ -f $oldlang/G.carpa ]; then
  oldlm=$oldlang/G.carpa
elif [ ! -f $oldlm ]; then
  echo "$0: expecting either $oldlang/G.fst or $oldlang/G.carpa to exist" &&\
    exit 1;
fi

echo "$0: using $oldlm as old LM"

[ ! -d $rnnlm_dir/rnnlm ] && echo "$0: Missing tf model folder $rnnlm_dir/rnnlm" && exit 1;

for f in $rnnlm_dir/unk.probs $oldlang/words.txt $indir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: Missing file $f" && exit 1
done

awk -v n=$0 -v w=$weight 'BEGIN {if (w < 0 || w > 1) {
  print n": Interpolation weight should be in the range of [0, 1]"; exit 1;}}' \
  || exit 1;

oldlm_command="fstproject --project_output=true $oldlm |"

mkdir -p $outdir/log
nj=`cat $indir/num_jobs` || exit 1;
cp $indir/num_jobs $outdir

oldlm_weight=`perl -e "print -1.0 * $weight;"`
if [ "$oldlm" == "$oldlang/G.fst" ]; then
  $cmd JOB=1:$nj $outdir/log/rescorelm.JOB.log \
    lattice-lmrescore --lm-scale=$oldlm_weight \
    "ark:gunzip -c $indir/lat.JOB.gz|" "$oldlm_command" ark:-  \| \
    lattice-lmrescore-tf-rnnlm --lm-scale=$weight \
    --max-ngram-order=$max_ngram_order \
    $rnnlm_dir/unk.probs $rnnlm_dir/wordlist.rnn.final $oldlang/words.txt ark:- "$rnnlm_dir/rnnlm" \
    "ark,t:|gzip -c>$outdir/lat.JOB.gz" || exit 1;
else
  $cmd JOB=1:$nj $outdir/log/rescorelm.JOB.log \
    lattice-lmrescore-const-arpa --lm-scale=$oldlm_weight \
    "ark:gunzip -c $indir/lat.JOB.gz|" "$oldlm" ark:-  \| \
    lattice-lmrescore-tf-rnnlm --lm-scale=$weight \
    --max-ngram-order=$max_ngram_order \
    $rnnlm_dir/unk.probs $rnnlm_dir/wordlist.rnn.final $oldlang/words.txt ark:- "$rnnlm_dir/rnnlm" \
    "ark,t:|gzip -c>$outdir/lat.JOB.gz" || exit 1;
fi
if ! $skip_scoring ; then
  err_msg="$0: Not scoring because local/score.sh does not exist or not executable."
  [ ! -x local/score.sh ] && echo $err_msg && exit 1;
  local/score.sh --cmd "$cmd" $data $oldlang $outdir
else
  echo "$0: Not scoring because --skip-scoring was specified."
fi

exit 0;

