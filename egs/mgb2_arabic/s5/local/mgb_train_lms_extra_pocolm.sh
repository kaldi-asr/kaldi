#!/usr/bin/env bash

# Copyright 2016  Johns Hopkins University (author: Daniel Povey)
#           2017  Vimal Manohar
# Apache 2.0
#
# It is based on the example scripts distributed with PocoLM

# It will first check if pocolm is installed and if not will process with installation

set -e
stage=0
set -o pipefail 
set -u

stage=0
dir=data/local/pocolm
cmd=run.pl

echo "$0 $@"  # Print the command line for logging
. utils/parse_options.sh || exit 1;

lm_text=$1
mer=$2

if [ $# -ne 2 ]; then
  echo "Usage: $0 <lm-text> <mer>"
  exit 1
fi

lm_dir=${dir}/data

mkdir -p $dir
. ./path.sh || exit 1;  # for KALDI_ROOT
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

num_dev_sentences=30000
RANDOM=0

if [ $stage -le 0 ]; then
  mkdir -p ${dir}/data
  mkdir -p ${dir}/data/text

  echo "$0: Getting the Data sources"

  rm ${dir}/data/text/* 2>/dev/null || true

  # Full acoustic transcripts
  cat data/train_mer$mer/text | cut -d ' ' -f 2- | \
    shuf > ${dir}/train_mer${mer}_text
  head -n $num_dev_sentences < ${dir}/train_mer${mer}_text > \
    ${dir}/data/text/dev.txt

  tail -n +$[num_dev_sentences+1] < ${dir}/train_mer${mer}_text | \
    gzip -c > \
    ${dir}/data/text/train_mer${mer}.txt.gz

  # Get text from the extra LM corpus
  cat $lm_text | gzip -c > ${dir}/data/text/mgb_arabic.txt.gz

  cp data/dev_non_overlap/text ${dir}/data/mgb2_dev.txt
fi

if [ $stage -le 1 ]; then
  mkdir -p $dir/data/work
  if [ ! -f $dir/data/work/word_counts/.done ]; then
    get_word_counts.py $dir/data/text $dir/data/work/word_counts
    touch $dir/data/work/word_counts/.done
  fi
fi

lexicon=data/local/dict/lexicon.txt 
[ ! -f $lexicon ] && echo "$0: No such file $lexicon" && exit 1;

if [ $stage -le 2 ]; then
  cat $lexicon | awk '{print $1}' > $dir/data/work/wordlist
  wordlist_to_vocab.py --unk-symbol="<UNK>" $dir/data/work/wordlist > \
    $dir/data/work/vocab_wordlist.txt
  touch $dir/data/work/.vocab_wordlist.txt.done
fi

order=4
wordlist=$dir/data/work/wordlist

min_counts="default=5 train_mer${mer}=2"
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
                 --fold-dev-into=train_mer$mer \
                 --min-counts="${min_counts}" \
                 ${dir}/data/text ${order} ${lm_dir}/work ${unpruned_lm_dir}

  for x in mgb2_dev; do
    $cmd ${unpruned_lm_dir}/log/compute_data_prob_${x}.log \
      get_data_prob.py ${dir}/data/${x}.txt ${unpruned_lm_dir} 

    cat ${unpruned_lm_dir}/log/compute_data_prob_${x}.log | grep -F '[perplexity'
  done
fi

if [ $stage -le 4 ]; then
  echo "$0: pruning the LM (to larger size)"
  # Using 20 million n-grams for a big LM for rescoring purposes.
  size=20000000
  $cmd ${dir}/data/lm_${order}_prune_big/log/prune_lm.log \
    prune_lm_dir.py --target-num-ngrams=$size --initial-threshold=0.02 \
    ${unpruned_lm_dir} ${dir}/data/lm_${order}_prune_big

  for x in mgb2_dev; do
    $cmd ${dir}/data/lm_${order}_prune_big/log/compute_data_prob_${x}.log \
      get_data_prob.py ${dir}/data/${x}.txt ${dir}/data/lm_${order}_prune_big

    cat ${dir}/data/lm_${order}_prune_big/log/compute_data_prob_${x}.log | grep -F '[perplexity'
  done

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

  for x in mgb2_dev; do
    $cmd ${dir}/data/lm_${order}_prune_small/log/compute_data_prob_${x}.log \
      get_data_prob.py ${dir}/data/${x}.txt ${dir}/data/lm_${order}_prune_small

    cat ${dir}/data/lm_${order}_prune_small/log/compute_data_prob_${x}.log | grep -F '[perplexity'
  done

  format_arpa_lm.py ${dir}/data/lm_${order}_prune_small | gzip -c > ${dir}/data/arpa/${order}gram_small.arpa.gz
fi
