#!/bin/bash

# Copyright 2017   Hainan Xu
# Apache 2.0

# This script rescores lattices with KALDI RNNLM trained on reversed text.
# The input directory should already be rescored with a forward RNNLM, preferably
# with the pruned algorithm, since smaller lattices make rescoring much faster.
# An example of the forward pruned rescoring is at
# egs/swbd/s5c/local/rnnlm/run_tdnn_lstm.sh
# One example script for backward RNNLM rescoring is at
# egs/swbd/s5c/local/rnnlm/run_tdnn_lstm_back.sh

# Begin configuration section.
cmd=run.pl
skip_scoring=false
max_ngram_order=4 # Approximate the lattice-rescoring by limiting the max-ngram-order
                  # if it's set, it merges histories in the lattice if they share
                  # the same ngram history and this prevents the lattice from 
                  # exploding exponentially. Details of the n-gram approximation
                  # method are described in section 2.3 of the paper
                  # http://www.danielpovey.com/files/2018_icassp_lattice_pruning.pdm

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
   echo "with Kaldi RNNLM trained on reversed text. See comments in file for details"
   echo ""
   echo "Usage: $0 [options] <old-lang-dir> <rnnlm-dir> \\"
   echo "                   <data-dir> <input-decode-dir> <output-decode-dir>"
   echo " e.g.: $0 data/lang_tg exp/rnnlm_lstm/ data/test \\"
   echo "                   exp/tri3/test_rnnlm_forward exp/tri3/test_rnnlm_bidirection"
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
if [ ! -f $oldlm ]; then
  echo "$0: file $oldlm not found; using $oldlang/G.carpa"
  oldlm=$oldlang/G.carpa
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

normalize_opt=
if $normalize; then
  normalize_opt="--normalize-probs=true"
fi
oldlm_command="fstproject --project_output=true $oldlm |"
special_symbol_opts=$(cat $rnnlm_dir/special_symbol_opts.txt)

word_embedding=
if [ -f $rnnlm_dir/word_embedding.final.mat ]; then
  word_embedding=$rnnlm_dir/word_embedding.final.mat
else
  word_embedding="'rnnlm-get-word-embedding $rnnlm_dir/word_feats.txt $rnnlm_dir/feat_embedding.final.mat -|'"
fi

mkdir -p $outdir/log
nj=`cat $indir/num_jobs` || exit 1;
cp $indir/num_jobs $outdir

# In order to rescore with a backward RNNLM, we first remove the original LM
# scores with lattice-lmrescore, before reversing the lattices
oldlm_weight=$(perl -e "print -1.0 * $weight;")
if [ "$oldlm" == "$oldlang/G.fst" ]; then
  $cmd JOB=1:$nj $outdir/log/rescorelm.JOB.log \
    lattice-lmrescore --lm-scale=$oldlm_weight \
    "ark:gunzip -c $indir/lat.JOB.gz|" "$oldlm_command" ark:-  \| \
    lattice-reverse ark:- ark:- \| \
    lattice-lmrescore-kaldi-rnnlm --lm-scale=$weight $special_symbol_opts \
    --max-ngram-order=$max_ngram_order $normalize_opt \
    $word_embedding "$rnnlm_dir/final.raw" ark:- ark:- \| \
    lattice-reverse ark:- "ark,t:|gzip -c>$outdir/lat.JOB.gz" || exit 1;
else
  $cmd JOB=1:$nj $outdir/log/rescorelm.JOB.log \
    lattice-lmrescore-const-arpa --lm-scale=$oldlm_weight \
    "ark:gunzip -c $indir/lat.JOB.gz|" "$oldlm" ark:-  \| \
    lattice-reverse ark:- ark:- \| \
    lattice-lmrescore-kaldi-rnnlm --lm-scale=$weight $special_symbol_opts \
    --max-ngram-order=$max_ngram_order $normalize_opt \
    $word_embedding "$rnnlm_dir/final.raw" ark:- ark:- \| \
    lattice-reverse ark:- "ark,t:|gzip -c>$outdir/lat.JOB.gz" || exit 1;
fi

if ! $skip_scoring ; then
  err_msg="$0: Not scoring because local/score.sh does not exist or not executable."
  [ ! -x local/score.sh ] && echo $err_msg && exit 1;
  echo local/score.sh --cmd "$cmd" $data $oldlang $outdir
  local/score.sh --cmd "$cmd" $data $oldlang $outdir
else
  echo "$0: Not scoring because --skip-scoring was specified."
fi

exit 0;
