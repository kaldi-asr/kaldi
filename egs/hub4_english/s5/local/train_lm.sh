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
dir=data/local/local_lm
cmd=run.pl
vocab_size=   # Preferred vocabulary size

echo "$0 $@"  # Print the command line for logging
. utils/parse_options.sh || exit 1;

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

num_dev_sentences=4500
RANDOM=0  # set seed for shuffling to ensure reproducibility

if [ $stage -le 0 ]; then
  mkdir -p ${dir}/data
  mkdir -p ${dir}/data/text

  echo "$0: Getting the Data sources"

  rm ${dir}/data/text/* 2>/dev/null || true

  # Take unique subset to make sure that the training text is not in the 
  # dev set.
  # Replace train with train_bn96 in order to use only the 1996 HUB4 set
  cat data/train/text | cut -d ' ' -f 2- | sort | uniq -c | \
    shuf > ${dir}/train_text_with_count
  head -n $num_dev_sentences < ${dir}/train_text_with_count | \
    awk '{for (i=0; i<$1; i++) {print $0;} }' | cut -d ' ' -f 2- > \
    ${dir}/data/text/dev.txt 
  tail -n +$[num_dev_sentences+1] < ${dir}/train_text_with_count | \
    awk '{for (i=0; i<$1; i++) {print $0;} }' | cut -d ' ' -f 2- > \
    ${dir}/data/text/train.txt

  # Get text from NA News corpus 
  for x in data/local/data/na_news/*; do
    y=`basename $x`
    [ -f $x/corpus.gz ] && ln -sf `readlink -f $x/corpus.gz` ${dir}/data/text/${y}.txt.gz
  done

  # Get text from 1996 CSR HUB4 LM corpus
  for x in `cat data/local/data/csr96_hub4/{train,test}.filelist`; do
    gunzip -c $x
  done | gzip -c > ${dir}/data/text/csr96_hub4.txt.gz
  
  # Get text from 1995 CSR-IV HUB4 corpus
  cat data/local/data/csr95_hub4/dev95_text \
    data/local/data/csr95_hub4/eval95_text \
    data/local/data/csr95_hub4/train95_text | cut -d ' ' -f 2- > \
    ${dir}/data/text/csr95_hub4.txt

  # Get text from NA News supplement corpus 
  for x in data/local/data/na_news_supp; do
    y=`basename $x`
    [ -f $x/corpus.gz ] && ln -sf `readlink -f $x/corpus.gz` ${dir}/data/text/${y}.txt.gz
  done

  # for reporting perplexities, we'll use the "real" dev set.
  # note, we can't put it in ${dir}/data/text/, because then pocolm would use
  # it as one of the data sources.
  for x in dev96pe dev96ue eval96 eval97 eval98 eval99_1 eval99_2; do
    cat data/$x/stm | awk '!/^;;/ {if (NF > 6) print $0}' | cut -d ' ' -f 1,7- | \
      awk '!/IGNORE_TIME_SEGMENT_IN_SCORING/ {print $0}' | \
      local/normalize_transcripts.pl "<NOISE>" "<SPOKEN_NOISE>" | \
      cut -d ' ' -f 2- > ${dir}/data/${x}.txt
  done
fi

if [ $stage -le 1 ]; then
  mkdir -p $dir/data/work
  if [ ! -f $dir/data/work/word_counts/.done ]; then
    get_word_counts.py $dir/data/text $dir/data/work/word_counts
    touch $dir/data/work/word_counts/.done
  fi
fi

if [ $stage -le 2 ]; then
  # decide on the vocabulary.

  # NA news corpus is not clean. So better not to get vocabulary from there.
  # for x in data/local/data/na_news/*; do
  #   y=$dir/data/work/word_counts/`basename $x`.counts
  #   [ -f $y ] && cat $y 
  # done | local/lm/merge_word_counts.py 15 > $dir/data/work/na_news.wordlist_counts

  cat $dir/data/work/word_counts/{train,dev}.counts | \
    local/lm/merge_word_counts.py 2 > $dir/data/work/train.wordlist_counts

  cat $dir/data/work/word_counts/csr96_hub4.counts | \
    local/lm/merge_word_counts.py 5 > $dir/data/work/csr96_hub4.wordlist_counts

  cat $dir/data/work/word_counts/csr95_hub4.counts | \
    local/lm/merge_word_counts.py 5 > $dir/data/work/csr95_hub4.wordlist_counts

  cat $dir/data/work/{train,csr96_hub4,csr95_hub4}.wordlist_counts | \
    perl -ane 'if ($F[1] =~ m/[A-Za-z]/) { print "$F[0] $F[1]\n"; }' | \
    local/lm/merge_word_counts.py 1 | sort -k 1,1nr > $dir/data/work/final.wordlist_counts

  if [ ! -z "$vocab_size" ]; then
    awk -v sz=$vocab_size 'BEGIN{count=-1;} 
    { i+=1; if (i == int(sz)) { count = $1; };
      if (count > 0 && count != $1) { exit(0); } 
      print $0;
    }' $dir/data/work/final.wordlist_counts
  else 
    cat $dir/data/work/final.wordlist_counts
  fi | awk '{print $2}' > $dir/data/work/wordlist
fi

order=4
wordlist=$dir/data/work/wordlist

min_counts='default=5 train=1 csr96_hub4=2,3 csr95_hub4=2,3'

lm_name="`basename ${wordlist}`_${order}"
if [ -n "${min_counts}" ]; then
  lm_name+="_`echo ${min_counts} | tr -s "[:blank:]" "_" | tr "," "." | tr "=" "-"`"
fi
unpruned_lm_dir=${lm_dir}/${lm_name}.pocolm

export PATH=$KALDI_ROOT/tools/pocolm/scripts:$PATH

if [ $stage -le 3 ]; then
  echo "$0: training the unpruned LM"

  $cmd ${unpruned_lm_dir}/log/train.log \
    train_lm.py  --wordlist=$wordlist --num-splits=10 --warm-start-ratio=20  \
                 --limit-unk-history=true \
                 --fold-dev-into=train \
                 --min-counts="${min_counts}" \
                 ${dir}/data/text ${order} ${lm_dir}/work ${unpruned_lm_dir}

  for x in dev96ue dev96pe eval96 eval97 eval98 eval99_1 eval99_2; do
    $cmd ${unpruned_lm_dir}/log/compute_data_prob_${x}.log \
      get_data_prob.py ${dir}/data/${x}.txt ${unpruned_lm_dir} 

    cat ${unpruned_lm_dir}/log/compute_data_prob_${x}.log | grep -F '[perplexity'
  done
  
  # train_lm.py: You can set --bypass-metaparameter-optimization='0.829,0.997,0.066,0.014,0.171,0.244,0.063,0.001,0.023,0.004,0.014,0.006,0.018,0.027,0.082,1.000,0.004,0.007,0.024,0.703,0.108,0.046,0.019,0.848,0.258,0.208,0.195,0.889,0.297,0.282,0.242' to get equivalent results
  # train_lm.py: Ngram counts: 98768 + 26286404 + 21077207 + 17945418 = 65407797
  
  # get_data_prob.py: log-prob of data/local/local_lm/data/dev96ue.txt given model data/local/local_lm/data/wordlist_4_default-5_train-1_csr96_hub4-2.3_csr95_hub4-2.3.pocolm was -4.88365261291 per word [perplexity = 132.112338899] over 18771.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/dev96pe.txt given model data/local/local_lm/data/wordlist_4_default-5_train-1_csr96_hub4-2.3_csr95_hub4-2.3.pocolm was -4.9299451353 per word [perplexity = 138.371920398] over 23710.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval96.txt given model data/local/local_lm/data/wordlist_4_default-5_train-1_csr96_hub4-2.3_csr95_hub4-2.3.pocolm was -4.8308081807 per word [perplexity = 125.312194639] over 20553.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval97.txt given model data/local/local_lm/data/wordlist_4_default-5_train-1_csr96_hub4-2.3_csr95_hub4-2.3.pocolm was -4.82377287988 per word [perplexity = 124.433679586] over 33234.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval98.txt given model data/local/local_lm/data/wordlist_4_default-5_train-1_csr96_hub4-2.3_csr95_hub4-2.3.pocolm was -4.88114977878 per word [perplexity = 131.782097071] over 33180.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval99_1.txt given model data/local/local_lm/data/wordlist_4_default-5_train-1_csr96_hub4-2.3_csr95_hub4-2.3.pocolm was -5.01175279868 per word [perplexity = 150.167719384] over 11529.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval99_2.txt given model data/local/local_lm/data/wordlist_4_default-5_train-1_csr96_hub4-2.3_csr95_hub4-2.3.pocolm was -5.01485733132 per word [perplexity = 150.634644387] over 16395.0 words.
  
fi

if [ $stage -le 4 ]; then
  echo "$0: pruning the LM (to larger size)"
  # Using 10 million n-grams for a big LM for rescoring purposes.
  size=10000000
  $cmd ${dir}/data/lm_${order}_prune_big/log/prune_lm.log \
    prune_lm_dir.py --target-num-ngrams=$size --initial-threshold=0.02 \
    ${unpruned_lm_dir} ${dir}/data/lm_${order}_prune_big

  for x in dev96ue dev96pe eval96 eval97 eval98 eval99_1 eval99_2; do
    $cmd ${dir}/data/lm_${order}_prune_big/log/compute_data_prob_${x}.log \
      get_data_prob.py ${dir}/data/${x}.txt ${dir}/data/lm_${order}_prune_big

    cat ${dir}/data/lm_${order}_prune_big/log/compute_data_prob_${x}.log | grep -F '[perplexity'
  done

  # get_data_prob.py: log-prob of data/local/local_lm/data/dev96ue.txt given model data/local/local_lm/data/lm_4_prune_big was -4.96695051249 per word [perplexity = 143.588348177] over 18771.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/dev96pe.txt given model data/local/local_lm/data/lm_4_prune_big was -5.01232680304 per word [perplexity = 150.253941052] over 23710.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval96.txt given model data/local/local_lm/data/lm_4_prune_big was -4.91227395027 per word [perplexity = 135.948202644] over 20553.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval97.txt given model data/local/local_lm/data/lm_4_prune_big was -4.92411302883 per word [perplexity = 137.567269311] over 33234.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval98.txt given model data/local/local_lm/data/lm_4_prune_big was -4.97443821579 per word [perplexity = 144.667530381] over 33180.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval99_1.txt given model data/local/local_lm/data/lm_4_prune_big was -5.10483206523 per word [perplexity = 164.816389804] over 11529.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval99_2.txt given model data/local/local_lm/data/lm_4_prune_big was -5.10905926136 per word [perplexity = 165.514575655] over 16395.0 words.

  mkdir -p ${dir}/data/arpa
  format_arpa_lm.py ${dir}/data/lm_${order}_prune_big | gzip -c > ${dir}/data/arpa/${order}gram_big.arpa.gz
fi

if [ $stage -le 5 ]; then
  echo "$0: pruning the LM (to smaller size)"
  # Using 2 million n-grams for a smaller LM for graph building.  Prune from the
  # bigger-pruned LM, it'll be faster.
  size=2000000
  
  $cmd ${dir}/data/lm_${order}_prune_small/log/prune_lm.log \
    prune_lm_dir.py --target-num-ngrams=$size ${dir}/data/lm_${order}_prune_big \
    ${dir}/data/lm_${order}_prune_small

  for x in dev96ue dev96pe eval96 eval97 eval98 eval99_1 eval99_2; do
    $cmd ${dir}/data/lm_${order}_prune_small/log/compute_data_prob_${x}.log \
      get_data_prob.py ${dir}/data/${x}.txt ${dir}/data/lm_${order}_prune_small

    cat ${dir}/data/lm_${order}_prune_small/log/compute_data_prob_${x}.log | grep -F '[perplexity'
  done

  # get_data_prob.py: log-prob of data/local/local_lm/data/dev96ue.txt given model data/local/local_lm/data/lm_4_prune_small was -5.12459372596 per word [perplexity = 168.105830741] over 18771.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/dev96pe.txt given model data/local/local_lm/data/lm_4_prune_small was -5.16866547448 per word [perplexity = 175.680231224] over 23710.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval96.txt given model data/local/local_lm/data/lm_4_prune_small was -5.08096906048 per word [perplexity = 160.929931226] over 20553.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval97.txt given model data/local/local_lm/data/lm_4_prune_small was -5.09222677679 per word [perplexity = 162.751870937] over 33234.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval98.txt given model data/local/local_lm/data/lm_4_prune_small was -5.12842796263 per word [perplexity = 168.751625556] over 33180.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval99_1.txt given model data/local/local_lm/data/lm_4_prune_small was -5.26755997571 per word [perplexity = 193.942161054] over 11529.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval99_2.txt given model data/local/local_lm/data/lm_4_prune_small was -5.27092234584 per word [perplexity = 194.595363921] over 16395.0 words

  format_arpa_lm.py ${dir}/data/lm_${order}_prune_small | gzip -c > ${dir}/data/arpa/${order}gram_small.arpa.gz
fi
