#!/bin/bash

# Copyright 2015  Guoguo Chen
#           2017  Hainan Xu
# Apache 2.0

# This script rescores lattices with KALDI RNNLM.
# It uses a simple n-gram approximation to limit the search space;
# A faster and more accurate way to rescore is at rnnlm/lmrescore_pruned.sh
# which is more prefered

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
normalize=false # If true, we add a normalization step to the output of the RNNLM
                # so that it adds up to *exactly* 1. Note that this is not necessary
                # as in our RNNLM setup, a properly trained network would automatically
                # have its normalization term close to 1. The details of this
                # could be found at http://www.danielpovey.com/files/2018_icassp_rnnlm.pdf

# End configuration section.

echo "$0 $@"  # Print the command line for logging

. ./utils/parse_options.sh

if [ $# != 5 ]; then
   echo "Does language model rescoring of lattices (remove old LM, add new LM)"
   echo "with Kaldi RNNLM."
   echo ""
   echo "Usage: $0 [options] <old-lang-dir> <rnnlm-dir> \\"
   echo "                   <data-dir> <input-decode-dir> <output-decode-dir>"
   echo " e.g.: $0 data/lang_tg exp/rnnlm_lstm/ data/test \\"
   echo "                   exp/tri3/test_tg exp/tri3/test_rnnlm_4gram"
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

[ ! -f $oldlm ] && echo "$0: Missing file $oldlm" && exit 1;
[ ! -f $rnnlm_dir/final.raw ] && echo "$0: Missing file $rnnlm_dir/final.raw" && exit 1;
[ ! -f $rnnlm_dir/feat_embedding.final.mat ] && [ ! -f $rnnlm_dir/word_embedding.final.mat ] && echo "$0: Missing word embedding file" && exit 1;

[ ! -f $oldlang/words.txt ] &&\
  echo "$0: Missing file $oldlang/words.txt" && exit 1;
! ls $indir/lat.*.gz >/dev/null &&\
  echo "$0: No lattices input directory $indir" && exit 1;
awk -v n=$0 -v w=$weight 'BEGIN {if (w < 0 || w > 1) {
  print n": Interpolation weight should be in the range of [0, 1]"; exit 1;}}' \
  || exit 1;

oldlm_command="fstproject --project_output=true $oldlm |"

special_symbol_opts=$(cat $rnnlm_dir/special_symbol_opts.txt)

word_embedding=
if [ -f $rnnlm_dir/word_embedding.final.mat ]; then
  word_embedding=$rnnlm_dir/word_embedding.final.mat
else
  word_embedding="'rnnlm-get-word-embedding $rnnlm_dir/word_feats.txt $rnnlm_dir/feat_embedding.final.mat -|'"
fi

normalize_opt=
if $normalize; then
  normalize_opt="--normalize-probs=true"
fi

mkdir -p $outdir/log
nj=$(cat $indir/num_jobs) || exit 1;
cp $indir/num_jobs $outdir

oldlm_weight=$(perl -e "print -1.0 * $weight;")
if [ "$oldlm" == "$oldlang/G.fst" ]; then
  $cmd JOB=1:$nj $outdir/log/rescorelm.JOB.log \
    lattice-lmrescore --lm-scale=$oldlm_weight \
    "ark:gunzip -c $indir/lat.JOB.gz|" "$oldlm_command" ark:-  \| \
    lattice-lmrescore-kaldi-rnnlm --lm-scale=$weight $special_symbol_opts \
    --max-ngram-order=$max_ngram_order $normalize_opt \
    $word_embedding "$rnnlm_dir/final.raw" ark:- \
    "ark,t:|gzip -c>$outdir/lat.JOB.gz" || exit 1;
else
  $cmd JOB=1:$nj $outdir/log/rescorelm.JOB.log \
    lattice-lmrescore-const-arpa --lm-scale=$oldlm_weight \
    "ark:gunzip -c $indir/lat.JOB.gz|" "$oldlm" ark:-  \| \
    lattice-lmrescore-kaldi-rnnlm --lm-scale=$weight $special_symbol_opts \
    --max-ngram-order=$max_ngram_order $normalize_opt \
    $word_embedding "$rnnlm_dir/final.raw" ark:- \
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
