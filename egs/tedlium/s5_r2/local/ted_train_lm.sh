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

num_dev_sentences=10000

bypass_metaparam_optim_opt=
# If you want to bypass the metaparameter optimization steps with specific metaparameters
# un-comment the following line, and change the numbers to some appropriate values.
# You can find the values from output log of train_lm.py.
# These example numbers of metaparameters is for 4-gram model (with min-counts)
# running with train_lm.py.
# The dev perplexity should be close to the non-bypassed model.
#bypass_metaparam_optim_opt="--bypass-metaparameter-optimization=0.794,0.022,0.761,0.083,0.043,0.013,1.000,0.359,0.123,0.078,1.000,0.613,0.227,0.227"
# Note: to use these example parameters, you may need to remove the .done files
# to make sure the make_lm_dir.py be called and tain only 3-gram model
#for order in 3; do
#rm -f ${lm_dir}/${num_word}_${order}.pocolm/.done

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
  
  # get wordlist
  awk '{print $1}' db/cantab-TEDLIUM/cantab-TEDLIUM.dct | sort | uniq > ${dir}/data/wordlist
fi

order=4

if [ $stage -le 1 ]; then  
  # decide on the vocabulary.                                                   
  # Note: you'd use --wordlist if you had a previously determined word-list     
  # that you wanted to use.                                                     
  # Note: if you have more than one order, use a certain amount of words as the
  # vocab and want to restrict max memory for 'sort', 
  # the following might be a more reasonable setting:                     
  # train_lm.py --num-word=${num_word} --num-splits=10 --warm-start-ratio=20 ${max_memory} \
  #             --min-counts='train=2 ted=1' \                               
  #             --keep-int-data=true --fold-dev-into=ted ${bypass_metaparam_optim_opt} \
  #             ${dir}/data/text ${order} ${lm_dir} 
  echo "$0: training the unpruned LM"
  train_lm.py  --wordlist=${dir}/data/wordlist --num-splits=10 --warm-start-ratio=20  \
               --fold-dev-into=ted ${bypass_metaparam_optim_opt} \
               --min-counts='train=2 ted=1' \
               ${dir}/data/text ${order} ${lm_dir}
  unpruned_lm_dir=${lm_dir}/wordlist_${order}.pocolm
  
  get_data_prob.py ${dir}/data/real_dev_set.txt ${unpruned_lm_dir} 2>&1 | grep -F '[perplexity'

  # currently (with min-counts), this is what we have:
  # get_data_prob.py: log-prob of data/local/local_lm/data/real_dev_set.txt given model data/local/local_lm/data/wordlist_4.pocolm was -5.13902242865 per word [perplexity = 170.51635206] over 18290.0 words.
  # before I added min-counts, this is what we had:
  # get_data_prob.py: log-prob of data/local/local_lm/data/real_dev_set.txt given model data/local/local_lm/data/lm_4 was -5.10576291033 per word [perplexity = 164.969879761] over 18290.0 words.
fi

if [ $stage -le 2 ]; then
  echo "$0: pruning the LM (to larger size)"
  # Using 10 million n-grams for a big LM for rescoring purposes.
  size=10000000
  prune_lm_dir.py --target-num-ngrams=$size --initial-threshold=0.02 ${unpruned_lm_dir} ${dir}/data/lm_${order}_prune_big

  get_data_prob.py ${dir}/data/real_dev_set.txt ${dir}/data/lm_${order}_prune_big 2>&1 | grep -F '[perplexity'
  
  # with min-counts: 
  # get_data_prob.py ${dir}/data/real_dev_set.txt ${dir}/data/lm_${order}_prune_big 2>&1 | grep -F '[perplexity'
  # get_data_prob.py: log-prob of data/local/local_lm/data/real_dev_set.txt given model data/local/local_lm/data/lm_4_prune_big was -5.17638942756 per word [perplexity = 176.963480425] over 18290.0 words.

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

  # with min-counts:
  # get_data_prob.py: log-prob of data/local/local_lm/data/real_dev_set.txt given model data/local/local_lm/data/lm_4_prune_small was -5.28346290049 per word [perplexity = 198.060250839] over 18290.0 words.
  # before adding min-counts:
  # get_data_prob.py: log-prob of data/local/local_lm/data/real_dev_set.txt given model data/local/local_lm/data/lm_4_prune_small was -5.27623197813 per word [perplexity = 195.631341646] over 18290.0 words.

  format_arpa_lm.py ${dir}/data/lm_${order}_prune_small | gzip -c > ${dir}/data/arpa/${order}gram_small.arpa.gz
fi

