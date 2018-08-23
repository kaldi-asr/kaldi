#!/usr/bin/env bash

# Copyright 2018  Xiaohui Zhang

# This script prepares a new rnnlm-dir commpatible with a new vocab from a provided word-list,
# given an exisiting rnnlm-dir containing a trained rnnlm. Basically, we copy the feature 
# embedding, a trained rnnlm and some config files from the old rnnlm-dir. And then we re-
# generate the unigram_probs.txt (a fixed unigram prob is assigned to words out of the orignal vocab),
# word_feats.txt and word embeddings.

cmd=run.pl
oov_unigram_prob=0.0000001

. utils/parse_options.sh

if [ $# != 3 ]; then
  echo "Usage: $0 [options] <word-list> <rnnlm-in-dir> <rnnlm-out-dir>"
  echo "Prepare a new directory <rnnlm-out-dir> with a given <word-list> and a valid <rnnlm-in-dir>."
  echo "  <word-list> is a vocabulary file with mapping to integers."
  exit 1
fi

set -e
[ -f path.sh ] && . ./path.sh

word_list=$1
rnnlm_in_dir=$2
rnnlm_out_dir=$3

for f in features.txt data_weights.txt oov.txt xconfig; do
  if [ ! -f $rnnlm_in_dir/config/$f ]; then
    echo "$0: file $rnnlm_in_dir/config/$f is not present."
    exit 1
  fi
done

for f in unigram_probs.txt feat_embedding.final.mat final.raw; do
  if [ ! -f $rnnlm_in_dir/$f ]; then
    echo "$0: file $rnnlm_in_dir/$f is not present."
    exit 1
  fi
done

echo "$0: Copying config directory."
mkdir -p $rnnlm_out_dir/config
for f in features.txt data_weights.txt oov.txt xconfig; do
  cp $rnnlm_in_dir/config/$f $rnnlm_out_dir/config
done

for f in feat_embedding.final.mat final.raw; do
  cp -L $rnnlm_in_dir/$f $rnnlm_out_dir/
done

echo "$0: Re-generating words.txt, unigram_probs.txt, word_feats.txt and word_embedding.final.mat."
cp $word_list $rnnlm_out_dir/config/words.txt

brk_id=`cat $rnnlm_out_dir/config/words.txt | wc -l`
echo "<brk> $brk_id" >> $rnnlm_out_dir/config/words.txt

# Generate new unigram_probs.txt. For words within the original vocab, we just take the prob
# from the original unigram_probs.txt. For new words added, we assign the prob as $oov_unigram_prob.
awk -v s=$rnnlm_in_dir/unigram_probs.txt -v t=$rnnlm_in_dir/config/words.txt  -v oov_prob=$oov_unigram_prob \
  'BEGIN { while ((getline<s) > 0) { id2prob[$1] = $2; } 
           while ((getline<t) > 0) { word2prob[$1] = id2prob[$2]; }
   } 
   { if ($1 in word2prob) print $2" "word2prob[$1]; else print $2" "oov_prob; }' \
   $rnnlm_out_dir/config/words.txt | sort -k1,1 -n > $rnnlm_out_dir/unigram_probs.txt

rnnlm/get_special_symbol_opts.py < $rnnlm_out_dir/config/words.txt > $rnnlm_out_dir/special_symbol_opts.txt

# Re-compute words_feats.txt and word embeddings.
rnnlm/get_word_features.py --unigram-probs=$rnnlm_out_dir/unigram_probs.txt --treat-as-bos='#0' \
  $rnnlm_out_dir/config/words.txt $rnnlm_out_dir/config/features.txt > $rnnlm_out_dir/word_feats.txt

rnnlm-get-word-embedding $rnnlm_out_dir/word_feats.txt $rnnlm_out_dir/feat_embedding.final.mat \
  $rnnlm_out_dir/word_embedding.final.mat
