#!/bin/bash

# Compute scores from RNNLM.  This script takes a directory
# $dir (e.g. dir=local/rnnlm/rnnlm.voc30.hl30 ),
# where it expects the files:
#  rnnlm  wordlist.rnn  unk.probs,
# and also an input file location where it can get the sentences to score, and
# an output file location to put the scores (negated logprobs) for each
# sentence.  This script uses the Kaldi-style "archive" format, so the input and
# output files will have a first field that corresponds to some kind of
# utterance-id or, in practice, utterance-id-1, utterance-id-2, etc., for the
# N-best list.
#
# Here, "wordlist.rnn" is the set of words, like a vocabulary,
# that the RNN was trained on (note, it won't include <s> or </s>),
# plus <RNN_UNK> which is a kind of class where we put low-frequency
# words; unk.probs gives the probs for words given this class, and it
# has, on each line, "word prob".

ensure_normalized_probs=false  # if true then we add the neccesary options to
                               # normalize the probabilities of RNNLM
                               # e.g. when using faster-rnnlm in the nce mode

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

rnnlm-sentence-probs $special_symbol_opts $dir/final.raw "$word_embedding" $tempdir/text.int > $tempdir/loglikes.rnn

[ $(cat $tempdir/loglikes.rnn | wc -l) -ne $(cat $tempdir/text | wc -l) ] && \
  echo "rnnlm rescoring failed" && exit 1;

paste $tempdir/loglikes.rnn | awk '{sum=0;for(i=1;i<=NF;i++)sum-=$i; print sum}' >$tempdir/scores

# scores out, with utterance-ids.
paste $tempdir/ids $tempdir/scores  > $scores_out


