#!/usr/bin/env bash

# Copyright 2016  Vincent Nguyen
#           2016  Johns Hopkins University (author: Daniel Povey)
#           2017  Ashish Arora
#           2017  Hossein Hadian
# Apache 2.0
#
# This script trains a LM on the training transcriptions and corpus text.
# It is based on the example scripts distributed with PocoLM

# It will check if pocolm is installed and if not will proceed with installation

set -e
stage=0
dir=data/local/local_lm
order=6
echo "$0 $@"  # Print the command line for logging
. ./utils/parse_options.sh || exit 1;

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

bypass_metaparam_optim_opt=
# If you want to bypass the metaparameter optimization steps with specific metaparameters
# un-comment the following line, and change the numbers to some appropriate values.
# You can find the values from output log of train_lm.py.
# These example numbers of metaparameters is for 4-gram model (with min-counts)
# running with train_lm.py.
# The dev perplexity should be close to the non-bypassed model.
#bypass_metaparam_optim_opt="--bypass-metaparameter-optimization=0.031,0.860,0.678,0.194,0.037,0.006,0.928,0.712,0.454,0.220,0.926,0.844,0.749,0.358,0.966,0.879,0.783,0.544,0.966,0.826,0.674,0.450"
# Note: to use these example parameters, you may need to remove the .done files
# to make sure the make_lm_dir.py be called and tain only 3-gram model
#for order in 3; do
#rm -f ${lm_dir}/${num_word}_${order}.pocolm/.done

if [ $stage -le 0 ]; then
  mkdir -p ${dir}/data
  mkdir -p ${dir}/data/text

  echo "$0: Getting the Data sources"

  rm ${dir}/data/text/* 2>/dev/null || true

  # use the validation data as the dev set.
  # Note: the name 'dev' is treated specially by pocolm, it automatically
  # becomes the dev set.

  cat data/local/text/cleaned/bpe_val.txt  > ${dir}/data/text/dev.txt
  # use the training data as an additional data source.
  # we can later fold the dev data into this.
  cat data/train/text | cut -d " " -f 2- >  ${dir}/data/text/train.txt
  cat data/local/text/cleaned/bpe_corpus.txt > ${dir}/data/text/corpus_text.txt
  # for reporting perplexities, we'll use the "real" dev set.
  # (the validation data is used as ${dir}/data/text/dev.txt to work
  # out interpolation weights.)
  # note, we can't put it in ${dir}/data/text/, because then pocolm would use
  # it as one of the data sources.
  cut -d " " -f 2-  < data/test/text  > ${dir}/data/real_dev_set.txt

  # get the wordlist from train and corpus text
  cat ${dir}/data/text/{train,corpus_text}.txt | tr '[:space:]' '[\n*]' | grep -v "^\s*$" | sort | uniq -c | sort -bnr > ${dir}/data/word_count
  cat ${dir}/data/word_count | awk '{print $2}' > ${dir}/data/wordlist
fi

if [ $stage -le 1 ]; then
  # decide on the vocabulary.
  # Note: you'd use --wordlist if you had a previously determined word-list
  # that you wanted to use.
  # Note: if you have more than one order, use a certain amount of words as the
  # vocab and want to restrict max memory for 'sort',
  echo "$0: training the unpruned LM"
  min_counts='train=1'
  wordlist=${dir}/data/wordlist

  lm_name="`basename ${wordlist}`_${order}"
  if [ -n "${min_counts}" ]; then
    lm_name+="_`echo ${min_counts} | tr -s "[:blank:]" "_" | tr "=" "-"`"
  fi
  unpruned_lm_dir=${lm_dir}/${lm_name}.pocolm
  train_lm.py  --wordlist=${wordlist} --num-splits=20 --warm-start-ratio=20 \
               --limit-unk-history=true \
               ${bypass_metaparam_optim_opt} \
               ${dir}/data/text ${order} ${lm_dir}/work ${unpruned_lm_dir}

  get_data_prob.py ${dir}/data/real_dev_set.txt ${unpruned_lm_dir} 2>&1 | grep -F '[perplexity'
  mkdir -p ${dir}/data/arpa
  format_arpa_lm.py ${unpruned_lm_dir} | gzip -c > ${dir}/data/arpa/${order}gram_unpruned.arpa.gz
fi

if [ $stage -le 2 ]; then
  echo "$0: pruning the LM (to larger size)"
  # Using 10 million n-grams for a big LM for rescoring purposes.
  size=10000000
  prune_lm_dir.py --target-num-ngrams=$size --initial-threshold=0.02 ${unpruned_lm_dir} ${dir}/data/lm_${order}_prune_big

  get_data_prob.py ${dir}/data/real_dev_set.txt ${dir}/data/lm_${order}_prune_big 2>&1 | grep -F '[perplexity'
  #[perplexity = 22.0613098868] over 151116.0 words
  mkdir -p ${dir}/data/arpa
  format_arpa_lm.py ${dir}/data/lm_${order}_prune_big | gzip -c > ${dir}/data/arpa/${order}gram_big.arpa.gz
fi

if [ $stage -le 3 ]; then
  echo "$0: pruning the LM (to smaller size)"
  # Using 2 million n-grams for a smaller LM for graph building.  Prune from the
  # bigger-pruned LM, it'll be faster.
  size=2000000
  prune_lm_dir.py --target-num-ngrams=$size ${dir}/data/lm_${order}_prune_big ${dir}/data/lm_${order}_prune_small

  get_data_prob.py ${dir}/data/real_dev_set.txt ${dir}/data/lm_${order}_prune_small 2>&1 | grep -F '[perplexity'
  #[perplexity = 23.4801171202] over 151116.0 words
  format_arpa_lm.py ${dir}/data/lm_${order}_prune_small | gzip -c > ${dir}/data/arpa/${order}gram_small.arpa.gz
fi
