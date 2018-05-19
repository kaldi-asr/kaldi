#!/bin/bash

# Copyright 2015  Guoguo Chen
#           2017  Hainan Xu
#           2018  Ke Li
# Apache 2.0

# This script rescores lattices with KALDI RNNLM.

# Begin configuration section.
cmd=run.pl
skip_scoring=false
max_ngram_order=4
acwt=0.1
N=10
weight=0.5  # Interpolation weight for RNNLM.
correction_weight=0.1
normalize=false
two_speaker_mode=true
one_best_mode=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

. ./utils/parse_options.sh

if [ $# != 7 ]; then
   echo Getting num-params = $#
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
utt2convo=$6
unigram=$7

oldlm=$oldlang/G.fst
carpa_option=
if [ ! -f $oldlm ]; then
  echo "$0: file $oldlm not found; looking for $oldlang/G.carpa"
  oldlm=$oldlang/G.carpa
  carpa_option="--use-const-arpa=true"
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

special_symbol_opts=$(cat $rnnlm_dir/special_symbol_opts.txt)

word_embedding=
if [ -f $rnnlm_dir/word_embedding.final.mat ]; then
  word_embedding=$rnnlm_dir/word_embedding.final.mat
else
  # word_embedding="'rnnlm-get-word-embedding $rnnlm_dir/word_feats.txt $rnnlm_dir/feat_embedding.final.mat -|'"
  word_embedding="rnnlm-get-word-embedding $rnnlm_dir/word_feats.txt $rnnlm_dir/feat_embedding.final.mat -|"
fi

mkdir -p $outdir/log
nj=`cat $indir/num_jobs` || exit 1;
cp $indir/num_jobs $outdir


$cmd JOB=1:$nj $outdir/log/rescorelm.JOB.log \
  lattice-lmrescore-kaldi-rnnlm-pruned-cache-adapt --lm-scale=$weight \
    $special_symbol_opts --acoustic-scale=$acwt \
    --max-ngram-order=$max_ngram_order $normalize_opt \
    --correction_weight=$correction_weight \
    --two-speaker-mode=$two_speaker_mode \
    --one-best-mode=$one_best_mode \
    $carpa_option $oldlm "$word_embedding" "$rnnlm_dir/final.raw" \
    "ark:gunzip -c $indir/lat.JOB.gz|" "ark,t:|gzip -c>$outdir/lat.JOB.gz" \
    $utt2convo $unigram || exit 1;

if ! $skip_scoring ; then
  err_msg="Not scoring because local/score.sh does not exist or not executable."
  [ ! -x local/score.sh ] && echo $err_msg && exit 1;
  local/score.sh --cmd "$cmd" $data $oldlang $outdir
else
  echo "Not scoring because requested so..."
fi

exit 0;
