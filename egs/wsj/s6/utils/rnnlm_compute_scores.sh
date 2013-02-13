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

. ./path.sh || exit 1;

rnnlm=$KALDI_ROOT/tools/rnnlm-0.3e/rnnlm

[ ! -f $rnnlm ] && echo No such program $rnnlm && exit 1;

if [ $# != 4 ]; then
  echo "Usage: rnnlm_compute_scores.sh <rnn-dir> <temp-dir> <input-text> <output-scores>"
  exit 1;
fi

dir=$1
tempdir=$2
text_in=$3
scores_out=$4

for x in rnnlm wordlist.rnn unk.probs; do
  if [ ! -f $dir/$x ]; then 
    echo "rnnlm_compute_scores.sh: expected file $dir/$x to exist."
    exit 1;
  fi
done

mkdir -p $tempdir
cat $text_in | awk '{for (x=2;x<=NF;x++) {printf("%s ", $x)} printf("\n");}' >$tempdir/text
cat $text_in | awk '{print $1}' > $tempdir/ids # e.g. utterance ids.
cat $tempdir/text | awk -v voc=$dir/wordlist.rnn -v unk=$dir/unk.probs \
  -v logprobs=$tempdir/loglikes.oov \
 'BEGIN{ while((getline<voc)>0) { invoc[$1]=1; } while ((getline<unk)>0){ unkprob[$1]=$2;} }
  { logprob=0; for (x=1;x<=NF;x++) { w=$x;  
    if (invoc[w]) { printf("%s ",w); } else {
      printf("<RNN_UNK> ");
      if (unkprob[w] != 0) { logprob += log(unkprob[w]); }
      else { print "Warning: unknown word ", w >"/dev/stderr"; logprob += log(1.0e-07); }}}
    printf("\n"); print logprob > logprobs } ' > $tempdir/text.nounk

# OK, now we compute the scores on the text with OOVs replaced
# with <RNN_UNK>

$rnnlm -independent -rnnlm $dir/rnnlm -test $tempdir/text.nounk -nbest -debug 0 | \
   awk '{print $1*log(10);}' > $tempdir/loglikes.rnn

[ `cat $tempdir/loglikes.rnn | wc -l` -ne `cat $tempdir/loglikes.oov | wc -l` ] && \
  echo "rnnlm rescoring failed" && exit 1;

paste $tempdir/loglikes.rnn $tempdir/loglikes.oov | awk '{print -($1+$2);}' >$tempdir/scores

# scores out, with utterance-ids.
paste $tempdir/ids $tempdir/scores  > $scores_out

