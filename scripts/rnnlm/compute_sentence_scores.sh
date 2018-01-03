#!/bin/bash

# This script is very similar to utils/rnnlm_compute_scores.sh, and it computes
# log-likelihoods from a Kaldi-RNNLM model instead of that of Mikolov's RNNLM.
# Because Kaldi-RNNLM uses letter-features which does not need an <OOS> symbol,
# we don't need the "unk.probs" file any more to add as a penalty term in sentence
# likelihoods.

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
cat $text_in | awk '{for (x=2;x<=NF;x++) {printf("%s ", $x)} printf("\n");}' >$tempdir/text
cat $text_in | awk '{print $1}' > $tempdir/ids # e.g. utterance ids.

cat $tempdir/text | sym2int.pl $dir/config/words.txt > $tempdir/text.int

special_symbol_opts=$(cat $dir/special_symbol_opts.txt)

rnnlm-sentence-probs --normalize-probs=$ensure_normalized_probs \
       $special_symbol_opts $dir/final.raw "$word_embedding" $tempdir/text.int > $tempdir/loglikes.rnn

[ $(cat $tempdir/loglikes.rnn | wc -l) -ne $(cat $tempdir/text | wc -l) ] && \
  echo "$0: rnnlm rescoring failed" && exit 1;

# We need the negative log-probabilities
paste $tempdir/loglikes.rnn | awk '{sum=0;for(i=1;i<=NF;i++)sum-=$i; print sum}' >$tempdir/scores

# Add utterance IDs
paste $tempdir/ids $tempdir/scores  > $scores_out


