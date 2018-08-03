#!/bin/bash

# Copyright 2016  Vincent Nguyen
#           2016  Johns Hopkins University (author: Daniel Povey)
#           2017  Ashish Arora
#           2017  Hossein Hadian
# Apache 2.0
#
# This script trains an LM on the LOB+Brown text data and IAM training transcriptions.
# It is based on the example scripts distributed with PocoLM

# It will check if pocolm is installed and if not will proceed with installation

set -e
stage=0
vocab_size=50000

echo "$0 $@"  # Print the command line for logging
. ./utils/parse_options.sh || exit 1;

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

bypass_metaparam_optim_opt=
# If you want to bypass the metaparameter optimization steps with specific metaparameters
# un-comment the following line, and change the numbers to some appropriate values.
# You can find the values from output log of train_lm.py.
# These example numbers of metaparameters is for 4-gram model (with min-counts)
# running with train_lm.py.
# The dev perplexity should be close to the non-bypassed model.
#bypass_metaparam_optim_opt=
# Note: to use these example parameters, you may need to remove the .done files
# to make sure the make_lm_dir.py be called and tain only 3-gram model
#for order in 3; do
#rm -f ${lm_dir}/${num_word}_${order}.pocolm/.done

if [ $stage -le 0 ]; then
  mkdir -p ${dir}/data
  mkdir -p ${dir}/data/text

  echo "$0: Getting the Data sources"

  rm ${dir}/data/text/* 2>/dev/null || true

  # Using LOB and brown corpus.
  cat data/local/lobcorpus/0167/download/LOB_COCOA/lob.txt | \
    local/remove_test_utterances_from_lob.py data/test/text data/val/text \
                                             > ${dir}/data/text/lob.txt
  cat data/local/browncorpus/brown.txt > ${dir}/data/text/brown.txt
  if [ -d "data/local/wellingtoncorpus" ]; then
    cat data/local/wellingtoncorpus/Wellington_annotation_removed.txt > ${dir}/data/text/wellington.txt
  fi

  # use the validation data as the dev set.
  # Note: the name 'dev' is treated specially by pocolm, it automatically
  # becomes the dev set.

  cat data/val/text | cut -d " " -f 2-  > ${dir}/data/text/dev.txt

  # use the training data as an additional data source.
  # we can later fold the dev data into this.
  cat data/train/text | cut -d " " -f 2- >  ${dir}/data/text/iam.txt

  # for reporting perplexities, we'll use the "real" dev set.
  # (the validation data is used as ${dir}/data/text/dev.txt to work
  # out interpolation weights.)
  # note, we can't put it in ${dir}/data/text/, because then pocolm would use
  # it as one of the data sources.
  cut -d " " -f 2-  < data/test/text  > ${dir}/data/real_dev_set.txt

  # get the wordlist from IAM text
  if [ -d "data/local/wellingtoncorpus" ]; then
    cat ${dir}/data/text/{iam,lob,brown,wellington}.txt | tr '[:space:]' '[\n*]' | grep -v "^\s*$" | sort | uniq -c | sort -bnr > ${dir}/data/word_count
  else
    echo "$0: Wellington Corpus not found. Proceeding without using that corpus."
    cat ${dir}/data/text/{iam,lob,brown}.txt | tr '[:space:]' '[\n*]' | grep -v "^\s*$" | sort | uniq -c | sort -bnr > ${dir}/data/word_count
  fi
  head -n $vocab_size ${dir}/data/word_count | awk '{print $2}' > ${dir}/data/wordlist
fi

order=3

if [ $stage -le 1 ]; then
  # decide on the vocabulary.
  # Note: you'd use --wordlist if you had a previously determined word-list
  # that you wanted to use.
  # Note: if you have more than one order, use a certain amount of words as the
  # vocab and want to restrict max memory for 'sort',
  echo "$0: training the unpruned LM"
  min_counts='brown=2 lob=2 iam=1'
  wordlist=${dir}/data/wordlist

  lm_name="`basename ${wordlist}`_${order}"
  if [ -n "${min_counts}" ]; then
    lm_name+="_`echo ${min_counts} | tr -s "[:blank:]" "_" | tr "=" "-"`"
  fi
  unpruned_lm_dir=${lm_dir}/${lm_name}.pocolm
  train_lm.py  --wordlist=${wordlist} --num-splits=10 --warm-start-ratio=20  \
               --limit-unk-history=true \
               ${bypass_metaparam_optim_opt} \
               ${dir}/data/text ${order} ${lm_dir}/work ${unpruned_lm_dir}

  get_data_prob.py ${dir}/data/real_dev_set.txt ${unpruned_lm_dir} 2>&1 | grep -F '[perplexity'
fi

if [ $stage -le 2 ]; then
  echo "$0: pruning the LM (to larger size)"
  # Using 1 million n-grams for a big LM for rescoring purposes.
  size=1000000
  prune_lm_dir.py --target-num-ngrams=$size --initial-threshold=0.02 ${unpruned_lm_dir} ${dir}/data/lm_${order}_prune_big

  get_data_prob.py ${dir}/data/real_dev_set.txt ${dir}/data/lm_${order}_prune_big 2>&1 | grep -F '[perplexity'

  mkdir -p ${dir}/data/arpa
  format_arpa_lm.py ${dir}/data/lm_${order}_prune_big | gzip -c > ${dir}/data/arpa/${order}gram_big.arpa.gz
fi

if [ $stage -le 3 ]; then
  echo "$0: pruning the LM (to smaller size)"
  # Using 500,000 n-grams for a smaller LM for graph building.  Prune from the
  # bigger-pruned LM, it'll be faster.
  size=500000
  prune_lm_dir.py --target-num-ngrams=$size ${dir}/data/lm_${order}_prune_big ${dir}/data/lm_${order}_prune_small

  get_data_prob.py ${dir}/data/real_dev_set.txt ${dir}/data/lm_${order}_prune_small 2>&1 | grep -F '[perplexity'

  format_arpa_lm.py ${dir}/data/lm_${order}_prune_small | gzip -c > ${dir}/data/arpa/${order}gram_small.arpa.gz
fi
