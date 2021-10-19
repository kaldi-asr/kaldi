#!/usr/bin/env bash

# Copyright 2021 ASLP, NWPU (Author: Hang Lyu)
#
# Apache 2.0

# In this shell script, we prepare an n-gram LM with the 'SRILM' toolkit.
# 1. Prepare the 'vocab' which is a wordlist without the silence word--'!SIL'.
#    And add <s> and </s>.
# 2. Remove the utt-id from the 'text' file and generate the training corpus.
# 3. Train an n-gram LM with the 'SRILM'.


heldout_set=10000
ngram_order=3


. ./path.sh
. ./utils/parse_options.sh


if [ $# -ne 3 ]; then
  echo "Usage: local/wenetspeech_train_lms.sh <lexicon> <word-segmented-text> <dir>"
  echo " e.g.: local/train_lms.sh data/local/dict/lexicon.txt data/corpus/text data/local/lm"
  echo ""
  echo " --heldout-set <numeric>  # the first <numeric> utterances is held for perplexity testing."
  echo " --ngram-order <order>    # N-gram order fro language model."
  exit 1;
fi

lexicon=$1
text=$2
dir=$3

for file in "$text" "$lexicon"; do
  [ ! -f $file ] && echo "$0: No such file $file" && exit 1;
done

ngram=`which ngram-count`
if [ -z $ngram ] || [ ! -x $ngram ]; then
  echo "$0: ngram-count is not found. Please use the script "
  echo "$0: tools/extras/install_srilm.sh to install it and set path.sh."
  exit 1
fi

mkdir -p $dir

# prepare the vocab
grep -w -v '!SIL' $lexicon | awk '{print $1}' |\
  cat - <(echo "<s>"; echo "</s>") > $dir/vocab

# prepare the LM training corpus and held-out set
cat $text | head -n $heldout_set |\
  awk '{ for(n=2; n<=NF; n++) {
         if (n<NF) { printf "%s ", $n;}
         else { print $n;} } }' > $dir/heldout_corpus

heldout_set_next=$((heldout_set + 1))
cat $text | tail -n +$heldout_set_next |\
  awk '{ for(n=2; n<=NF; n++) {
         if (n<NF) { printf "%s ", $n;}
         else { print $n;} } }' > $dir/train_corpus

# train
lm=$dir/lm_${ngram_order}gram.arpa.gz
$ngram -order $ngram_order -vocab $dir/vocab -unk -text $dir/train_corpus \
  -map-unk "<UNK>" -kndiscount -interpolate -lm $lm

ngram -lm $lm -ppl $dir/heldout_corpus

echo "$0: Done"
exit 0
