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
cmd=run.pl

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

if [ $stage -le 0 ]; then
  mkdir -p ${dir}/data
  mkdir -p ${dir}/data/text

  echo "$0: Getting the Data sources"

  rm ${dir}/data/text/* 2>/dev/null || true

  # Unzip TEDLIUM 6 data sources, normalize apostrophe+suffix to previous word, gzip the result.
  gunzip -c db/TEDLIUM_release2/LM/*.en.gz | sed 's/ <\/s>//g' | \
    local/join_suffix.py | awk '{print "foo "$0}' | \
    local/normalize_transcript.pl '<NOISE>' | cut -d ' ' -f 2- | gzip -c  > ${dir}/data/text/train.txt.gz
  # use a subset of the annotated training data as the dev set .
  # Note: the name 'dev' is treated specially by pocolm, it automatically
  # becomes the dev set.
  head -n $num_dev_sentences < data/train/text | cut -d " " -f 2-  > ${dir}/data/text/dev.txt
  # .. and the rest of the training data as an additional data source.
  # we can later fold the dev data into this.
  tail -n +$[$num_dev_sentences+1] < data/train/text | cut -d " " -f 2- >  ${dir}/data/text/ted.txt

  cat data/train_si284/text | cut -d " " -f 2- > ${dir}/data/text/wsj_si284.txt

  # for reporting perplexities, we'll use the "real" dev set.
  # (a subset of the training data is used as ${dir}/data/text/ted.txt to work
  # out interpolation weights.
  # note, we can't put it in ${dir}/data/text/, because then pocolm would use
  # it as one of the data sources.
  cut -d " " -f 2-  < data/dev/text  > ${dir}/data/real_dev_set.txt
fi

if [ $stage -le 1 ]; then
  mkdir -p $dir/data/work
  get_word_counts.py $dir/data/text $dir/data/work/word_counts
  touch $dir/data/work/word_counts/.done
fi

if [ $stage -le 2 ]; then
  # decide on the vocabulary.
  
  cat $dir/data/work/word_counts/{ted,dev}.counts | \
    local/lm/merge_word_counts.py 2 > $dir/data/work/ted.wordlist_counts

  cat $dir/data/work/word_counts/train.counts | \
    local/lm/merge_word_counts.py 5 > $dir/data/work/train.wordlist_counts

  cat $dir/data/work/word_counts/wsj_si284.counts | \
    local/lm/merge_word_counts.py 2 > $dir/data/work/wsj_si284.wordlist_counts

  cat $dir/data/work/{ted,train,wsj_si284}.wordlist_counts | \
    perl -ane 'if ($F[1] =~ m/[A-Za-z]/) { print "$F[0] $F[1]\n"; }' | \
    local/lm/merge_word_counts.py 1 | sort -k 1,1nr > $dir/data/work/final.wordlist_counts

  if [ ! -z "$vocab_size" ]; then
    awk -v sz=$vocab_size 'BEGIN{count=-1;} 
    { i+=1; 
      if (i == int(sz)) { 
        count = $1; 
      };
      if (count > 0 && count != $1) { 
        exit(0); 
      } 
      print $0;
    }' $dir/data/work/final.wordlist_counts
  else 
    cat $dir/data/work/final.wordlist_counts
  fi | awk '{print $2}' > $dir/data/work/wordlist
fi

order=4
wordlist=${dir}/data/work/wordlist
min_counts='train=2 ted=1 wsj_si284=5'

# Uncomment these if you want to remove WSJ data from LM. It should not 
# affect much. WSJ data improves perplexity by a couple of points.
# min_counts='train=2 ted=1'
# [ -f $dir/data/text/wsj_si284.txt ] && mv $dir/data/text/wsj_si284.txt $dir/data/
# [ -f $dir/data/work/word_counts/wsj_si284.counts ] && mv $dir/data/work/word_counts/wsj_si284.counts $dir/data/work

lm_name="`basename ${wordlist}`_${order}"
if [ -n "${min_counts}" ]; then
  lm_name+="_`echo ${min_counts} | tr -s "[:blank:]" "_" | tr "," "." | tr "=" "-"`"
fi
unpruned_lm_dir=${lm_dir}/${lm_name}.pocolm

if [ $stage -le 3 ]; then
  echo "$0: training the unpruned LM"

  $cmd ${unpruned_lm_dir}/log/train.log \
    train_lm.py  --wordlist=${wordlist} --num-splits=10 --warm-start-ratio=20  \
                 --limit-unk-history=true \
                 --fold-dev-into=ted ${bypass_metaparam_optim_opt} \
                 --min-counts="${min_counts}" \
                 ${dir}/data/text ${order} ${lm_dir}/work ${unpruned_lm_dir}

  for x in real_dev_set; do
    $cmd ${unpruned_lm_dir}/log/compute_data_prob_${x}.log \
      get_data_prob.py ${dir}/data/${x}.txt ${unpruned_lm_dir} 

    cat ${unpruned_lm_dir}/log/compute_data_prob_${x}.log | grep -F '[perplexity'
  done
  # Preplexity with just cantab-tedlium LM and Ted text: [perplexity = 157.87] over 18290.0 words
  # Perplexity with WSJ text added:
  # log-prob of data/local/local_lm/data/real_dev_set.txt given model data/local/local_lm/data/wordlist_4_train-2_ted-1_wsj_si284-5.pocolm was -5.05607815615 per word [perplexity = 156.973681282] over 18290.0 words.

fi

if [ $stage -le 4 ]; then
  echo "$0: pruning the LM (to larger size)"
  # Using 10 million n-grams for a big LM for rescoring purposes.
  size=10000000
  $cmd ${dir}/data/lm_${order}_prune_big/log/prune_lm.log \
    prune_lm_dir.py --target-num-ngrams=$size --initial-threshold=0.02 ${unpruned_lm_dir} ${dir}/data/lm_${order}_prune_big

  for x in real_dev_set; do 
    $cmd ${dir}/data/lm_${order}_prune_big/log/compute_data_prob_${x}.log \
      get_data_prob.py ${dir}/data/${x}.txt ${dir}/data/lm_${order}_prune_big

    cat ${dir}/data/lm_${order}_prune_big/log/compute_data_prob_${x}.log | grep -F '[perplexity'
  done  

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
  $cmd ${dir}/data/lm_${order}_prune_small/log/prune_lm.log \
    prune_lm_dir.py --target-num-ngrams=$size ${dir}/data/lm_${order}_prune_big ${dir}/data/lm_${order}_prune_small

  for x in real_dev_set; do
    $cmd ${dir}/data/lm_${order}_prune_small/log/compute_data_prob_${x}.log \
      get_data_prob.py ${dir}/data/${x}.txt ${dir}/data/lm_${order}_prune_small

    cat ${dir}/data/lm_${order}_prune_small/log/compute_data_prob_${x}.log | grep -F '[perplexity'
  done

  # current results, after adding --limit-unk-history=true (needed for modeling OOVs and not blowing up LG.fst):
  # get_data_prob.py: log-prob of data/local/local_lm/data/real_dev_set.txt given model data/local/local_lm/data/lm_4_prune_small was -5.29432352378 per word [perplexity = 199.202824404 over 18290.0 words.

  format_arpa_lm.py ${dir}/data/lm_${order}_prune_small | gzip -c > ${dir}/data/arpa/${order}gram_small.arpa.gz
fi
