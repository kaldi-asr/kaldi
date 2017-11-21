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
RANDOM=0

if [ $stage -le 0 ]; then
  mkdir -p ${dir}/data
  mkdir -p ${dir}/data/text

  echo "$0: Getting the Data sources"

  rm ${dir}/data/text/* 2>/dev/null || true

  # Take unique subset to make sure that the training text is not in the 
  # dev set.
  cat data/train_bn96/text | cut -d ' ' -f 2- | sort | uniq -c | \
    shuf > ${dir}/train_bn96_text_with_count
  head -n $num_dev_sentences < ${dir}/train_bn96_text_with_count | \
    awk '{for (i=0; i<$1; i++) {print $0;} }' | cut -d ' ' -f 2- > \
    ${dir}/data/text/dev.txt 
  tail -n +$[num_dev_sentences+1] < ${dir}/train_bn96_text_with_count | \
    awk '{for (i=0; i<$1; i++) {print $0;} }' | cut -d ' ' -f 2- > \
    ${dir}/data/text/train_bn96.txt

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

  # # Get text from NA News supplement corpus 
  # for x in data/local/data/na_news/*; do
  #   y=`basename $x`
  #   [ -f $x/corpus.gz ] && ln -sf `readlink -f $x/corpus.gz` ${dir}/data/text/${y}.txt.gz
  # done

  # for reporting perplexities, we'll use the "real" dev set.
  # note, we can't put it in ${dir}/data/text/, because then pocolm would use
  # it as one of the data sources.
  for x in dev96pe dev96ue eval96 eval97 eval98 eval99_1 eval99_2; do
    cat data/$x/stm | awk '!/^;;/ {if (NF > 6) print $0}' | cut -d ' ' -f 1,7- | \
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

  cat $dir/data/work/word_counts/{train_bn96,dev}.counts | \
    local/lm/merge_word_counts.py 2 > $dir/data/work/train_bn96.wordlist_counts

  cat $dir/data/work/word_counts/csr96_hub4.counts | \
    local/lm/merge_word_counts.py 5 > $dir/data/work/csr96_hub4.wordlist_counts

  cat $dir/data/work/word_counts/csr95_hub4.counts | \
    local/lm/merge_word_counts.py 5 > $dir/data/work/csr95_hub4.wordlist_counts

  cat $dir/data/work/{train_bn96,csr96_hub4,csr95_hub4}.wordlist_counts | \
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
wordlist=$dir/data/work/wordlist

min_counts='default=5 train_bn96=1 csr96_hub4=2,3 csr95_hub4=2,3'

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
                 --fold-dev-into=train_bn96 \
                 --min-counts="${min_counts}" \
                 ${dir}/data/text ${order} ${lm_dir}/work ${unpruned_lm_dir}

  for x in dev96ue dev96pe eval96 eval97 eval98 eval99_1 eval99_2; do
    $cmd ${unpruned_lm_dir}/log/compute_data_prob_${x}.log \
      get_data_prob.py ${dir}/data/${x}.txt ${unpruned_lm_dir} 

    cat ${unpruned_lm_dir}/log/compute_data_prob_${x}.log | grep -F '[perplexity'
  done
  
  # train_lm.py: Ngram counts: 190742 + 31139856 + 14766071 + 13851899 = 59948568
  # train_lm.py: You can set --bypass-metaparameter-optimization='1.000,0.007,0.000,0.002,0.000,0.006,0.003,0.000,0.000,0.000,0.001,0.002,0.002,0.000,0.000,0.000,0.003,0.000,0.000,0.604,0.187,0.044,0.012,1.000,0.490,0.026,0.001,0.732,0.328,0.281,0.218' to get equivalent results
  # get_data_prob.py: log-prob of data/local/local_lm_bn_nanews_csr96/data/real_dev_set.txt given model data/local/local_lm_bn_nanews_csr96/data/wordlist_4_default-5_bn-1.pocolm was -4.9927348506 per word [perplexity = 147.338822662] over 33180.0 words.

  # get_data_prob.py: log-prob of data/local/local_lm/data/dev96pe.txt given model data/local/local_lm/data/wordlist_4_default-5_train_bn96-1_csr96_hub4-2.3_csr95_hub4-2.3.pocolm was -4.92985727862 per word [perplexity = 138.359764034] over 23760.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/dev96ue.txt given model data/local/local_lm/data/wordlist_4_default-5_train_bn96-1_csr96_hub4-2.3_csr95_hub4-2.3.pocolm was -4.88171588624 per word [perplexity = 131.85672102] over 18821.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval96.txt given model data/local/local_lm/data/wordlist_4_default-5_train_bn96-1_csr96_hub4-2.3_csr95_hub4-2.3.pocolm was -4.85089075845 per word [perplexity = 127.85422637] over 20625.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval97.txt given model data/local/local_lm/data/wordlist_4_default-5_train_bn96-1_csr96_hub4-2.3_csr95_hub4-2.3.pocolm was -4.84370861758 per word [perplexity = 126.939248987] over 33340.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval98.txt given model data/local/local_lm/data/wordlist_4_default-5_train_bn96-1_csr96_hub4-2.3_csr95_hub4-2.3.pocolm was -4.91000862327 per word [perplexity = 135.640584068] over 33180.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval99_1.txt given model data/local/local_lm/data/wordlist_4_default-5_train_bn96-1_csr96_hub4-2.3_csr95_hub4-2.3.pocolm was -5.03738768271 per word [perplexity = 154.067016944] over 11529.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval99_2.txt given model data/local/local_lm/data/wordlist_4_default-5_train_bn96-1_csr96_hub4-2.3_csr95_hub4-2.3.pocolm was -5.02574438024 per word [perplexity = 152.283570813] over 16395.0 words.

fi
  for x in dev96ue dev96pe eval96 eval97 eval98 eval99_1 eval99_2; do
    $cmd ${unpruned_lm_dir}/log/compute_data_prob_${x}.log \
      get_data_prob.py ${dir}/data/${x}.txt ${unpruned_lm_dir} 

    cat ${unpruned_lm_dir}/log/compute_data_prob_${x}.log | grep -F '[perplexity'
  done

  

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

  # get_data_prob.py data/local/local_lm_bn_nanews_csr96/data/real_dev_set.txt data/local/local_lm_bn_nanews_csr96/data/lm_4_prune_big
  # grep -F '[perplexity'
  # get_data_prob.py: log-prob of data/local/local_lm_bn_nanews_csr96/data/real_dev_set.txt given model data/local/local_lm_bn_nanews_csr96/data/lm_4_prune_big was -5.05700399638 per word [perplexity = 157.11908113]
  # over 33180.0 words.

  # get_data_prob.py: log-prob of data/local/local_lm/data/dev96pe.txt given model data/local/local_lm/data/lm_4_prune_big was -5.00197658249 per word [perplexity = 148.706800062] over 23760.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/dev96ue.txt given model data/local/local_lm/data/lm_4_prune_big was -4.95522131024 per word [perplexity = 141.914009921] over 18821.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval96.txt given model data/local/local_lm/data/lm_4_prune_big was -4.91668501333 per word [perplexity = 136.54920329] over 20625.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval97.txt given model data/local/local_lm/data/lm_4_prune_big was -4.92810468806 per word [perplexity = 138.117488385] over 33340.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval98.txt given model data/local/local_lm/data/lm_4_prune_big was -4.98326999699 per word [perplexity = 145.950861062] over 33180.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval99_1.txt given model data/local/local_lm/data/lm_4_prune_big was -5.10923357186 per word [perplexity = 165.543429098] over 11529.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval99_2.txt given model data/local/local_lm/data/lm_4_prune_big was -5.10475193474 per word [perplexity = 164.803183515] over 16395.0 words.

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

  # get_data_prob.py data/local/local_lm_bn_nanews_csr96/data/real_dev_set.txt data/local/local_lm_bn_nanews_csr96/data/lm_4_prune_small
  # grep -F '[perplexity'
  # get_data_prob.py: log-prob of data/local/local_lm_bn_nanews_csr96/data/real_dev_set.txt given model data/local/local_lm_bn_nanews_csr96/data/lm_4_prune_small was -5.27172473478 per word [perplexity = 194.751567749] over 33180.0 words.
  # float-counts-to-pre-arpa: output [ 190743 673670 802551 351512 ] n-grams

  # get_data_prob.py: log-prob of data/local/local_lm/data/dev96pe.txt given model data/local/local_lm/data/lm_4_prune_small was -5.15402161616 per word [perplexity = 173.126339858] over 23760.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/dev96ue.txt given model data/local/local_lm/data/lm_4_prune_small was -5.10689797354 per word [perplexity = 165.157237313] over 18821.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval96.txt given model data/local/local_lm/data/lm_4_prune_small was -5.07740442667 per word [perplexity = 160.357296176] over 20625.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval97.txt given model data/local/local_lm/data/lm_4_prune_small was -5.09747614277 per word [perplexity = 163.608461382] over 33340.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval98.txt given model data/local/local_lm/data/lm_4_prune_small was -5.13563068716 per word [perplexity = 169.971484911] over 33180.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval99_1.txt given model data/local/local_lm/data/lm_4_prune_small was -5.26596417642 per word [perplexity = 193.632915104] over 11529.0 words.
  # get_data_prob.py: log-prob of data/local/local_lm/data/eval99_2.txt given model data/local/local_lm/data/lm_4_prune_small was -5.26092885453 per word [perplexity = 192.660361662] over 16395.0 words.

  format_arpa_lm.py ${dir}/data/lm_${order}_prune_small | gzip -c > ${dir}/data/arpa/${order}gram_small.arpa.gz
fi

