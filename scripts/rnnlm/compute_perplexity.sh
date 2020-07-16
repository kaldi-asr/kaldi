#!/usr/bin/env bash

# This script computes perplexity of text on the specified RNNLM model. 

[ -f ./path.sh ] && . ./path.sh

use_gpu=no
. utils/parse_options.sh

if [ $# != 2 ]; then
  echo "Usage: $0 [options] <rnn-dir> <input-text>"
  echo "Options: "
  echo "  --use_gpu (yes|no|optional|wait)  # whether to use gpu [no]."
  exit 1
fi

dir=$1
text_in=$2

# the format of the $text_in file is one sentence per line, without explicit
# <s> or </s> symbols, and without utterance-id's, for example:

# ====== begin file ======
# well western new york is supposed to be used to this kind of weather but
# yeah you are right
# in um anaheim california you know just
# ====== end file ======

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

special_symbol_opts=$(cat $dir/special_symbol_opts.txt)

opt="--normalize-probs=true --use-gpu=${use_gpu}"

ppl=$(rnnlm-sentence-probs ${opt} \
       $special_symbol_opts $dir/final.raw "$word_embedding" \
       <(cat $text_in | sym2int.pl $dir/config/words.txt | awk '{print "utt_id ", $0}') | \
       awk '{for(i=2;i<=NF;i++) a+=$i; b+=NF-1}END{print exp(-a / b)}')

echo "$0: perplexity is $ppl"
