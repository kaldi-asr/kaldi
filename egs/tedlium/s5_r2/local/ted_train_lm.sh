#!/bin/bash

# Copyright 2016  Vincent Nguyen
#           2016  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0
#
# This script trains a LM on the Cantab-Tedlium text data and tedlium acoustic training data.
# It is based on the example scripts distributed with PocoLM

# It will first check if pocolm is installed and if not will process with installation
# It will then get the source data from the pre-downloaded Cantab-Tedlium files
# and the pre-prepared data/train text source.


set -e

echo "$0 $@"  # Print the command line for logging
. utils/parse_options.sh || exit 1;

dir=data/local/local_lm

mkdir -p $dir
. ./path.sh || exit 1; # for KALDI_ROOT
export PATH=$KALDI_ROOT/tools/pocolm/scripts:$PATH
( # First make sure the pocolm toolkit is installed.
 cd $KALDI_ROOT/tools || exit 1;
 if [ -d pocolm ]; then
   echo Not installing the pocolm toolkit since it is already there.
 else
   echo "Please install the PocoLM toolkit with kaldi/tools/install_pocolm.sh"
   exit 1;
 fi
) || exit 1;

num_dev_sentences=15000

if [ ! -d ${dir}/data ]; then
  mkdir ${dir}/data
  mkdir ${dir}/data/text
fi

echo Getting the Data sources
head -n $num_dev_sentences < db/cantab-TEDLIUM/cantab-TEDLIUM.txt | sed 's/ <\/s>//g'  > ${dir}/data/text/dev.txt
cat db/cantab-TEDLIUM/cantab-TEDLIUM.txt | sed 's/ <\/s>//g'  > ${dir}/data/text/train.txt
cat data/train/text | cut -d " " -f 2- > ${dir}/data/text/ted.txt

echo Getting word counts
get_word_counts.py ${dir}/data/text ${dir}/data/word_counts

# decide on the vocabulary.
echo Preparing vocabulary
awk '{print $1}' db/cantab-TEDLIUM/cantab-TEDLIUM.dct | sort | uniq > ${dir}/data/wordlist
wordlist_to_vocab.py ${dir}/data/wordlist > ${dir}/data/vocab.txt

echo Preparing Int_data
prepare_int_data.py ${dir}/data/text ${dir}/data/vocab.txt ${dir}/data/int

echo Starting the LM process
for order in 3 4; do
  get_counts.py ${dir}/data/int ${order} ${dir}/data/counts_${order}
  ratio=20
  splits=10
  subset_count_dir.sh ${dir}/data/counts_${order} ${ratio} ${dir}/data/counts_${order}_subset${ratio}

  mkdir -p ${dir}/data/optimize_${order}_subset${ratio}

  optimize_metaparameters.py --progress-tolerance=1.0e-05 --num-splits=${splits} \
    ${dir}/data/counts_${order}_subset${ratio} ${dir}/data/optimize_${order}_subset${ratio}

  optimize_metaparameters.py --warm-start-dir=${dir}/data/optimize_${order}_subset${ratio} \
    --progress-tolerance=1.0e-03 --num-splits=${splits} \
    ${dir}/data/counts_${order} ${dir}/data/optimize_${order}

  make_lm_dir.py --num-splits=${splits} ${dir}/data/counts_${order} \
     ${dir}/data/optimize_${order}/final.metaparams ${dir}/data/lm_${order}

  mkdir -p ${dir}/data/arpa
  format_arpa_lm.py ${dir}/data/lm_${order} | gzip -c > ${dir}/data/arpa/${vocab_size}_${order}gram_unpruned.arpa.gz

  get_data_prob.py ${dir}/data/text/dev.txt ${dir}/data/lm_${order} 2>&1 | grep -F '[perplexity'

done

# pruning the LM for order 3 only, and using 3400000 n-grams to get a 30MB LM size
order=3
size=3400000
    prune_lm_dir.py --target-num-ngrams=$size ${dir}/data/lm_${order} ${dir}/data/lm_${order}_prune 2>&1 | tail -n 5 | head -n 3
    get_data_prob.py ${dir}/data/text/dev.txt ${dir}/data/lm_${order}_prune 2>&1 | grep -F '[perplexity'

    format_arpa_lm.py ${dir}/data/lm_${order}_prune | gzip -c > ${dir}/data/arpa/${vocab_size}_${order}gram_prune.arpa.gz


