#!/bin/bash
# Copyright 2021  Xiaomi Corporation (Author: Yongqing Wang)
#                 Seasalt AI, Inc (Author: Guoguo Chen)

# This script rains a typical ngram language model.

set -e -o pipefail

stage=0
lm_order=4
vocab_size=50000000      # Cap the vocabulary so that it won't blow up.
cmd=run.pl
mem=10G

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. ./utils/parse_options.sh || exit 1;


if [[ $# -ne 2 ]]; then
  echo "Usage: $0: <lm-corpus> <output-lm-dir>"
  echo " e.g.: $0: data/local/lm/corpus.txt data/local/lm/"
  echo "Options:"
  echo "  --lm-order <order>    # N-gram order for language model."
  echo "  --vocab-size <size>   # Cap for vocabulary size."
  exit 1
fi

lm_corpus=$1
lm_dir=$2

word_counts=$lm_dir/word_counts.txt
vocab=$lm_dir/vocab.txt
full_corpus=$lm_dir/lm-norm.txt.gz
lm=$lm_dir/lm_${lm_order}gram.arpa.gz

mkdir -p $lm_dir
if [ "$stage" -le 1 ]; then
  echo "$0: Creating the corpus and the vocabulary"
  # The following sequence of commands does the following:
  # 1) Eliminates duplicate sentences and saves the resulting corpus
  # 2) Splits the corpus into words
  # 3) Sorts the words in respect to their frequency
  # 4) Caps the vocabulary to $vocab_size words, sorted by their frequencies
  # 5) Saves an alphabetically sorted vocabulary, that include the most frequent
  #    $vocab_size words
  cat $lm_corpus |\
    sort -u | tee >(gzip >$full_corpus) |\
    tr -s '[[:space:]]' '\n' | sort | uniq -c | sort -k1 -n -r |\
    head -n $vocab_size | tee $word_counts |\
    awk '{print $2}' | sort >$vocab || exit 1;
  echo "$0: Word counts saved to $word_counts"
  echo "$0: Vocabulary saved as $vocab"
  echo "$0: All unique sentences (in sorted order) stored in $full_corpus"
  echo "$0: Counting the total number word tokens in the corpus ..."
  echo "$0: There are $(wc -w < <(zcat $full_corpus)) tokens in the corpus"
fi

if [ "$stage" -le 2 ]; then
  echo "$0: Training a N-gram language model"
  ngram=`which ngram-count`
  if [ -z $ngram ] || [ ! -x $ngram ]; then
    echo "$0: Please install SRILM and set path.sh accordingly."
    exit 1;
  fi

  mkdir -p $lm_dir/log || exit 1;
  $cmd --mem $mem JOB=1:1 $lm_dir/log/ngram.JOB.log \
    $ngram -order $lm_order -interpolate -unk -map-unk \""<UNK>"\" \
    -limit-vocab -vocab $vocab -text $full_corpus -lm $lm || exit 1;
  du -h $lm
fi

echo "$0: Done"
