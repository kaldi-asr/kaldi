#!/bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0
#
# This script trains a LM on the Broadcast News transcripts.
# It is based on the example scripts distributed with PocoLM.

# It will first check if pocolm is installed and if not will process with installation


set -e
set -o pipefail 
set -u

stage=0

echo "$0 $@"  # Print the command line for logging
. utils/parse_options.sh || exit 1;

dir=data/local/local_lm
lm_dir=${dir}/data

mkdir -p $dir
. ./path.sh || exit 1; # for KALDI_ROOT
export PATH=$KALDI_ROOT/tools/pocolm/scripts:$PATH
( # First make sure the pocolm toolkit is installed.
 cd $KALDI_ROOT/tools || exit 1;
 if [ -d pocolm ]; then
   echo Not installing the pocolm toolkit since it is already there.
 else
   echo "$0: Please install the PocoLM toolkit with: "
   echo " cd ../../../tools; extras/install_pocolm.sh; cd -"
   exit 1;
 fi
) || exit 1;

num_dev_sentences=5000
RANDOM=0

if [ $stage -le 0 ]; then
  mkdir -p ${dir}/data
  mkdir -p ${dir}/data/text

  echo "$0: Getting the Data sources"

  rm ${dir}/data/text/* 2>/dev/null || true

  cat data/train/text | shuf > ${dir}/train_text
  head -n $num_dev_sentences < ${dir}/train_text | cut -d ' ' -f 2- > ${dir}/data/text/dev.txt 
  tail -n +$[num_dev_sentences+1] < ${dir}/train_text | cut -d ' ' -f 2- > ${dir}/data/text/bn.txt

  for x in data/local/data/na_news/*; do
    y=`basename $x`
    [ -f $x/corpus.gz ] && ln -sf `readlink -f $x/corpus.gz` ${dir}/data/text/${y}.txt.gz
  done

  # for reporting perplexities, we'll use the "real" dev set.
  # (a subset of the training data is used as ${dir}/data/text/ted.txt to work
  # out interpolation weights.
  # note, we can't put it in ${dir}/data/text/, because then pocolm would use
  # it as one of the data sources.
  cat data/eval98/stm | awk '!/^;;/ {if (NF > 6) print $0}' | cut -d ' ' -f 1,7- | \
    local/normalize_transcripts.pl "<NOISE>" "<SPOKEN_NOISE>" | \
    cut -d ' ' -f 2- > ${dir}/data/real_dev_set.txt
fi

if [ $stage -le 1 ]; then
  mkdir -p $dir/data/work
  if [ ! -f $dir/data/work/word_counts/.done ]; then
    get_word_counts.py $dir/data/text $dir/data/work/word_counts
    touch $dir/data/work/word_counts/.done
  fi
fi

if [ $stage -le 2 ]; then
  for x in data/local/data/na_news/*; do
    y=$dir/data/work/word_counts/`basename $x`.counts
    [ -f $y ] && cat $y 
  done | local/lm/merge_word_counts.py 15 > $dir/data/work/na_news.wordlist_counts

  cat $dir/data/work/word_counts/{bn,dev}.counts | \
    local/lm/merge_word_counts.py 2 > $dir/data/work/bn.wordlist_counts

  cat $dir/data/work/na_news.wordlist_counts $dir/data/work/bn.wordlist_counts | \
    perl -ane 'if ($F[1] =~ m/[A-Za-z]/) { print "$F[1]\n"; }' | \
    sort -u > $dir/data/work/wordlist
fi

order=4
wordlist=$dir/data/work/wordlist

min_counts='default=5 bn=1'

lm_name="`basename ${wordlist}`_${order}"
if [ -n "${min_counts}" ]; then
  lm_name+="_`echo ${min_counts} | tr -s "[:blank:]" "_" | tr "=" "-"`"
fi
unpruned_lm_dir=${lm_dir}/${lm_name}.pocolm

if [ $stage -le 3 ]; then
  # decide on the vocabulary.
  # Note: if you have more than one order, use a certain amount of words as the
  # vocab and want to restrict max memory for 'sort',
  echo "$0: training the unpruned LM"
  train_lm.py  --wordlist=$wordlist --num-splits=10 --warm-start-ratio=20  \
               --limit-unk-history=true \
               --fold-dev-into=bn \
               --min-counts="${min_counts}" \
               ${dir}/data/text ${order} ${lm_dir}/work ${unpruned_lm_dir}

  get_data_prob.py ${dir}/data/real_dev_set.txt ${unpruned_lm_dir} 2>&1 | grep -F '[perplexity'
  #[perplexity = 157.87] over 18290.0 words
  
  mkdir -p ${dir}/data/arpa
  format_arpa_lm.py ${unpruned_lm_dir} | gzip -c > ${dir}/data/arpa/${order}gram.arpa.gz
fi

if [ $stage -le 4 ]; then
  echo "$0: pruning the LM (to larger size)"
  # Using 10 million n-grams for a big LM for rescoring purposes.
  size=10000000
  prune_lm_dir.py --target-num-ngrams=$size --initial-threshold=0.02 ${unpruned_lm_dir} ${dir}/data/lm_${order}_prune_big

  get_data_prob.py ${dir}/data/real_dev_set.txt ${dir}/data/lm_${order}_prune_big 2>&1 | grep -F '[perplexity'

  # current results, after adding --limit-unk-history=true:
  # get_data_prob.py: log-prob of data/local/local_lm/data/real_dev_set.txt given model data/local/local_lm/data/lm_4_prune_big was -5.16562818753 per word [perplexity = 175.147449465] over 18290.0 words.


  mkdir -p ${dir}/data/arpa
  format_arpa_lm.py ${dir}/data/lm_${order}_prune_big | gzip -c > ${dir}/data/arpa/${order}gram_big.arpa.gz
fi

if [ $stage -le 5 ]; then
  echo "$0: pruning the LM (to smaller size)"
  # Using 2 million n-grams for a smaller LM for graph building.  Prune from the
  # bigger-pruned LM, it'll be faster.
  size=2000000
  prune_lm_dir.py --target-num-ngrams=$size ${dir}/data/lm_${order}_prune_big ${dir}/data/lm_${order}_prune_small

  get_data_prob.py ${dir}/data/real_dev_set.txt ${dir}/data/lm_${order}_prune_small 2>&1 | grep -F '[perplexity'

  # current results, after adding --limit-unk-history=true (needed for modeling OOVs and not blowing up LG.fst):
  # get_data_prob.py: log-prob of data/local/local_lm/data/real_dev_set.txt given model data/local/local_lm/data/lm_4_prune_small was -5.29432352378 per word [perplexity = 199.202824404 over 18290.0 words.

  format_arpa_lm.py ${dir}/data/lm_${order}_prune_small | gzip -c > ${dir}/data/arpa/${order}gram_small.arpa.gz
fi

