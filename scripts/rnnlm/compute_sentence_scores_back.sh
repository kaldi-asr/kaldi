#!/usr/bin/env bash

# Copyright 2017  Hainan Xu
#           2017  Szu-Jui Chen

# This script is very similar to rnnlm/compute_sentence_scores.sh, where it do the
# same procedure for reversed data. And it computes log-likelihoods from a
# Kaldi-RNNLM model instead of that of Mikolov's RNNLM. Because Kaldi-RNNLM uses
# letter-features which does not need an <OOS> symbol, we don't need the "unk.probs"
# file any more to add as a penalty term in sentence likelihoods.

ensure_normalized_probs=false  # If true then the probabilities computed by the
                               # RNNLM will be correctly normalized. Note it is
                               # OK to set it to false because Kaldi-RNNLM is
                               # trained in a way that ensures the sum of probabilities
                               # is close to 1.

. ./path.sh || exit 1;
. utils/parse_options.sh

if [ $# != 4 ]; then
  echo "Usage: $0 <rnn-dir> <temp-dir> <input-text> <output-scores>"
  exit 1;
fi

dir=$1
tempdir=$2
text_in=$3
scores_out=$4

if [ -f $dir/word_embedding.final.mat ]; then
  word_embedding=$dir/word_embedding.final.mat
else
  [ ! -f $dir/feat_embedding.final.mat ] &&
             echo "$0: expect file $dir/feat_embedding.final.mat to exit"
  word_embedding="rnnlm-get-word-embedding $dir/word_feats.txt $dir/feat_embedding.final.mat -|"
fi

for x in final.raw config/words.txt; do
  if [ ! -f $dir/$x ]; then 
    echo "$0: expected file $dir/$x to exist."
    exit 1;
  fi
done

mkdir -p $tempdir
cat $text_in | sym2int.pl -f 2- $dir/config/words.txt | \
    awk '{printf("%s ",$1);for(i=NF;i>1;i--) printf("%s ",$i); print""}' > $tempdir/text.int
    
special_symbol_opts=$(cat ${dir}/special_symbol_opts.txt)

rnnlm-sentence-probs --normalize-probs=$ensure_normalized_probs \
       $special_symbol_opts $dir/final.raw "$word_embedding" $tempdir/text.int > $tempdir/loglikes.rnn
# Now $tempdir/loglikes.rnn has the following structure
# utt-id log P(word1 | <s>) log P(word2 | <s> word1) ... log P(</s> | all word histories)
# for example,
#
# en_4156-A_058697-058813-2 -3.57205 -2.70411 -4.29876 -3.63707 -6.00299 -2.11093 -2.03955
# en_4156-A_058697-058813-3 -6.6074 -1.21244 -3.89991 -3.23747 -5.35102 -1.90448 -1.77809
# en_4156-A_058697-058813-4 -5.09022 -1.24148 -4.76337 -4.75594 -5.77118 -2.08555 -2.18403
# en_4156-A_058697-058813-5 -4.54489 -2.97485 -3.93646 -3.28041 -5.18779 -2.83356 -1.72601
# en_4156-A_058697-058813-6 -2.31464 -3.74738 -4.03309 -3.22942 -5.66818 -2.0396 -1.64734
# en_4156-A_058697-058813-7 -5.0728 -2.96303 -4.6539 -3.20266 -5.40682 -2.10625 -1.90956

[ $(cat $tempdir/loglikes.rnn | wc -l) -ne $(cat $tempdir/text.int | wc -l) ] && \
  echo "$0: rnnlm rescoring failed" && exit 1;
  
# We need the negative log-probabilities
cat $tempdir/loglikes.rnn | awk '{sum=0;for(i=2;i<=NF;i++)sum-=$i; print $1,sum}' >$scores_out
