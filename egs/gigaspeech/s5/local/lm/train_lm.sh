#!/bin/bash
# Copyright 2014 Vassil Panayotov
#           2021 Xiaomi Corporation Yongqing Wang
# Apache 2.0

. ./cmd.sh || exit 1
. ./path.sh || exit 1


# how many words we want in the LM's vocabulary
vocab_size=50000000

# LM pruning threshold for the 'small' trigram model
prune_thresh_small=0.0000003

# LM pruning threshold for the 'medium' trigram model
prune_thresh_medium=0.0000001

# how many text normalization jobs to run in parallel
if [[ $# -ne 2 ]]; then
  echo "Usage: $1 <corpus.txt> <out-lm-dir>"
  echo "where,"
  echo "  <corpus.txt>: normalized txt" 
  echo "  <out-lm-dir>: the directory to store the trained ARPA model"
  exit 1
fi

stage=0
corpus=$1
lm_dir=$2
order=$3
word_counts=$lm_dir/word_counts.txt
vocab=$lm_dir/vocab.txt
full_corpus=$lm_dir/lm-norm.txt.gz
gram_lm=$lm_dir/lm_${order}gram.arpa.gz
if [ "$stage" -le 1 ]; then
  echo "Selecting the vocabulary ($vocab_size words) ..."
  mkdir -p $lm_dir
  echo "Making the corpus and the vocabulary ..."
  # The following sequence of commands does the following:
  # 1) Eliminates duplicate sentences and saves the resulting corpus
  # 2) Splits the corpus into words
  # 3) Sorts the words in respect to their frequency
  # 4) Saves the list of the first $vocab_size words sorted by their frequencies
  # 5) Saves an alphabetically sorted vocabulary, that include the most frequent $vocab_size words
  cat $corpus | sort -u | tee >(gzip >$full_corpus) | tr -s '[[:space:]]' '\n' | sort | uniq -c | sort -k1 -n -r |head -n $vocab_size | tee $word_counts | awk '{print $2}' | sort >$vocab || exit 1
  echo "Word counts saved to '$word_counts'"
  echo "Vocabulary saved as '$vocab'"
  echo "All unique sentences (in sorted order) stored in '$full_corpus'"
  echo "Counting the total number word tokens in the corpus ..."
  echo "There are $(wc -w < <(zcat $full_corpus)) tokens in the corpus"
fi

if [ "$stage" -le 2 ]; then
  echo "Training a 4-gram LM ..."
  command -v ngram-count 1>/dev/null 2>&1 || { echo "Please install SRILM and set path.sh accordingly"; exit 1; }
  echo "This implementation assumes that you have a lot of free RAM(> 12GB) on your machine"
  echo "If that's not the case, consider something like: http://joshua-decoder.org/4.0/large-lms.html"
  ngram-count -order $order  -interpolate \
    -unk -map-unk "<UNK>" -limit-vocab -vocab $vocab -text $full_corpus -lm $gram_lm || exit 1
  du -h $gram_lm
fi

exit 0
