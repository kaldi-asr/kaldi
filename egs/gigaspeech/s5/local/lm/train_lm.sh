#!/bin/bash
# Copyright 2014 Vassil Panayotov
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
word_counts=$lm_dir/word_counts.txt
vocab=$lm_dir/gigaspeech-vocab.txt
full_corpus=$lm_dir/lm-norm.txt.gz

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

trigram_lm=$lm_dir/lm_tgram.arpa.gz

if [ "$stage" -le 2 ]; then
  echo "Training a 3-gram LM ..."
  command -v ngram-count 1>/dev/null 2>&1 || { echo "Please install SRILM and set path.sh accordingly"; exit 1; }
  echo "This implementation assumes that you have a lot of free RAM(> 12GB) on your machine"
  echo "If that's not the case, consider something like: http://joshua-decoder.org/4.0/large-lms.html"
  ngram-count -order 3  -interpolate \
    -unk -map-unk "<UNK>" -limit-vocab -vocab $vocab -text $full_corpus -lm $trigram_lm || exit 1
  du -h $trigram_lm
fi

exit 0

trigram_pruned_small=$lm_dir/lm_tgsmall.arpa.gz

if [ "$stage" -le 5 ]; then
  echo "Creating a 'small' pruned 3-gram LM (threshold: $prune_thresh_small) ..."
  command -v ngram 1>/dev/null 2>&1 || { echo "Please install SRILM and set path.sh accordingly"; exit 1; }
  ngram -prune $prune_thresh_small -lm $trigram_lm -write-lm $trigram_pruned_small || exit 1
  du -h $trigram_pruned_small
fi

trigram_pruned_medium=$lm_dir/lm_tgmed.arpa.gz

if [ "$stage" -le 5 ]; then
  echo "Creating a 'medium' pruned 3-gram LM (threshold: $prune_thresh_medium) ..."
  command -v ngram 1>/dev/null 2>&1 || { echo "Please install SRILM and set path.sh accordingly"; exit 1; }
  ngram -prune $prune_thresh_medium -lm $trigram_lm -write-lm $trigram_pruned_medium || exit 1
  du -h $trigram_pruned_medium
fi

fourgram_lm=$lm_dir/lm_fglarge.arpa.gz

if [ "$stage" -le 4 ]; then
  # This requires even more RAM than the 3-gram
  echo "Training a 4-gram LM ..."
  command -v ngram-count 1>/dev/null 2>&1 || { echo "Please install SRILM and set path.sh accordingly"; exit 1; }
  ngram-count -order 4  -kndiscount -interpolate \
    -unk -map-unk "<UNK>" -limit-vocab -vocab $vocab -text $full_corpus -lm $fourgram_lm || exit 1
  du -h $fourgram_lm
fi
exit 0
