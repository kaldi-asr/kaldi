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
stage=0

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
   echo "$0: Please install the PocoLM toolkit with: "
   echo " cd ../../../tools; extras/install_pocolm.sh; cd -"
   exit 1;
 fi
) || exit 1;

num_dev_sentences=10000


if [ $stage -le 0 ]; then
  mkdir -p ${dir}/data
  mkdir -p ${dir}/data/text

  echo "$0: Getting the Data sources"

  rm ${dir}/data/text/* 2>/dev/null || true

  # cantab-TEDLIUM is the larger data source.  gzip it.
  sed 's/ <\/s>//g' < db/cantab-TEDLIUM/cantab-TEDLIUM.txt | gzip -c  > ${dir}/data/text/train.txt.gz
  # use a subset of the annotated training data as the dev set .
  # Note: the name 'dev' is treated specially by pocolm, it automatically
  # becomes the dev set.
  head -n $num_dev_sentences < data/train/text | cut -d " " -f 2-  > ${dir}/data/text/dev.txt
  # .. and the rest of the training data as an additional data source.
  # we can later fold the dev data into this.
  tail -n +$[$num_dev_sentences+1] < data/train/text | cut -d " " -f 2- >  ${dir}/data/text/ted.txt

  # for reporting perplexities, we'll use the "real" dev set.
  # (a subset of the training data is used as ${dir}/data/text/ted.txt to work
  # out interpolation weights.
  # note, we can't put it in ${dir}/data/text/, because then pocolm would use
  # it as one of the data sources.
  cut -d " " -f 2-  < data/dev/text  > ${dir}/data/real_dev_set.txt
fi

if [ $stage -le 1 ]; then
  echo "$0: Getting word counts"
  get_word_counts.py ${dir}/data/text ${dir}/data/word_counts

  # decide on the vocabulary.
  echo Preparing vocabulary
  awk '{print $1}' db/cantab-TEDLIUM/cantab-TEDLIUM.dct | sort | uniq > ${dir}/data/wordlist
  wordlist_to_vocab.py ${dir}/data/wordlist > ${dir}/data/vocab.txt
fi

if [ $stage -le 2 ]; then
  echo "$0: Preparing integerized data"
  prepare_int_data.py ${dir}/data/text ${dir}/data/vocab.txt ${dir}/data/int
fi


order=4

if [ $stage -le 3 ]; then
  echo "$0: getting counts"
  # the LM will be very large unless we eliminate singleton counts from the
  # cantab-TEDLIUM data source, which is large (this is called 'train'),
  # but leave singletons from the TEDLIUM transcripts ('ted').
  get_counts.py --min-counts='train=2 ted=1' ${dir}/data/int ${order} ${dir}/data/counts_${order}
fi

if [ $stage -le 4 ]; then
  echo "$0: building the LM"
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
    --fold-dev-into=ted ${dir}/data/optimize_${order}/final.metaparams ${dir}/data/lm_${order}

  get_data_prob.py ${dir}/data/real_dev_set.txt ${dir}/data/lm_${order} 2>&1 | grep -F '[perplexity'

  # currently (with min-counts), this is what we have:
  # get_data_prob.py: log-prob of data/local/local_lm/data/real_dev_set.txt given model data/local/local_lm/data/lm_4 was -5.13902242865 per word [perplexity = 170.548963022] over 18290.0 words.
  # before I added min-counts, this is what we had:
  # get_data_prob.py: log-prob of data/local/local_lm/data/real_dev_set.txt given model data/local/local_lm/data/lm_4 was -5.10576291033 per word [perplexity = 164.969879761] over 18290.0 words.
fi

if [ $stage -le 5 ]; then
  echo "$0: pruning the LM (to larger size)"
  # Using 10 million n-grams for a big LM for rescoring purposes.
  size=10000000
  prune_lm_dir.py --target-num-ngrams=$size --initial-threshold=0.02 ${dir}/data/lm_${order} ${dir}/data/lm_${order}_prune_big

  # get_data_prob.py ${dir}/data/real_dev_set.txt ${dir}/data/lm_${order}_prune_big 2>&1 | grep -F '[perplexity'
  # get_data_prob.py: log-prob of data/local/local_lm/data/real_dev_set.txt given model data/local/local_lm/data/lm_4_prune_big was -5.17638942756 per word [perplexity = 177.042431097] over 18290.0 words.

  mkdir -p ${dir}/data/arpa
  format_arpa_lm.py ${dir}/data/lm_${order}_prune_big | gzip -c > ${dir}/data/arpa/${order}gram_big.arpa.gz
fi

if [ $stage -le 6 ]; then
  echo "$0: pruning the LM (to smaller size)"
  # Using 2 million n-grams for a smaller LM for graph building.  Prune from the
  # bigger-pruned LM, it'll be faster.
  size=2000000
  prune_lm_dir.py --target-num-ngrams=$size ${dir}/data/lm_${order}_prune_big ${dir}/data/lm_${order}_prune_small

  get_data_prob.py ${dir}/data/real_dev_set.txt ${dir}/data/lm_${order}_prune_small 2>&1 | grep -F '[perplexity'

  # currently:
  # get_data_prob.py: log-prob of data/local/local_lm/data/real_dev_set.txt given model data/local/local_lm/data/lm_4_prune_small was -5.28346290049 per word [perplexity = 197.051063452] over 18290.0 words.
  # before adding min-counts:
  # get_data_prob.py: log-prob of data/local/local_lm/data/real_dev_set.txt given model data/local/local_lm/data/lm_4_prune_small was -5.27623197813 per word [perplexity = 195.631341646] over 18290.0 words.

  format_arpa_lm.py ${dir}/data/lm_${order}_prune_small | gzip -c > ${dir}/data/arpa/${order}gram_small.arpa.gz
fi



