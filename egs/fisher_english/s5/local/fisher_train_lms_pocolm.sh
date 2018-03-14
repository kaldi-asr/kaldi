#!/bin/bash

# Copyright 2016  Vincent Nguyen
#           2016  Johns Hopkins University (author: Daniel Povey)
#           2017  Vimal Manohar
# Apache 2.0
#
# This script is used to train LMs using pocolm toolkit. 
# We use limit-unk-history=true, which truncates the history left of OOV word.
# This ensure the graph is compact when using phone LM to model OOV word.
# See the script local/run_unk_model.sh.

set -e
stage=0

text=data/train/text
lexicon=data/local/dict/lexicon.txt
dir=data/local/pocolm

num_ngrams_large=5000000
num_ngrams_small=2500000

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

for f in "$text" "$lexicon"; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

num_dev_sentences=10000

if [ $stage -le 0 ]; then
  mkdir -p ${dir}/data
  mkdir -p ${dir}/data/text

  echo "$0: Getting the Data sources"

  rm ${dir}/data/text/* 2>/dev/null || true

  cleantext=$dir/text_all.gz

  cut -d ' ' -f 2- $text | awk -v lex=$lexicon '
  BEGIN{
    while((getline<lex) >0) { seen[$1]=1; }
  }
  {
    for(n=1; n<=NF;n++) {  
      if (seen[$n]) { 
        printf("%s ", $n); 
      } else {
        printf("<unk> ");
      } 
    }
    printf("\n");
  }' | gzip -c > $cleantext || exit 1;

  # This is for reporting perplexities
  gunzip -c $dir/text_all.gz | head -n $num_dev_sentences > \
    ${dir}/data/test.txt

  # use a subset of the annotated training data as the dev set .
  # Note: the name 'dev' is treated specially by pocolm, it automatically
  # becomes the dev set.
  gunzip -c $dir/text_all.gz | tail -n +$[num_dev_sentences+1] | \
    head -n $num_dev_sentences > ${dir}/data/text/dev.txt

  gunzip -c $dir/text_all.gz | tail -n +$[2*num_dev_sentences+1] > \
    ${dir}/data/text/train.txt

  # for reporting perplexities, we'll use the "real" dev set.
  # (a subset of the training data is used as ${dir}/data/text/dev.txt to work
  # out interpolation weights.
  # note, we can't put it in ${dir}/data/text/, because then pocolm would use
  # it as one of the data sources.
  cat data/dev/text data/test/text | cut -d " " -f 2- > ${dir}/data/real_dev_set.txt

  cat $lexicon | awk '{print $1}' | sort | uniq  | awk '
  {
    if ($1 == "<s>") {
      print "<s> is in the vocabulary!" | "cat 1>&2"
      exit 1;
    }
    if ($1 == "</s>") {
      print "</s> is in the vocabulary!" | "cat 1>&2"
      exit 1;
    }
    printf("%s\n", $1);
  }' > $dir/data/wordlist || exit 1;
fi
  
order=4
wordlist=${dir}/data/wordlist

lm_name="`basename ${wordlist}`_${order}"
min_counts='train=1'
if [ -n "${min_counts}" ]; then
  lm_name+="_`echo ${min_counts} | tr -s "[:blank:]" "_" | tr "=" "-"`"
fi

unpruned_lm_dir=${lm_dir}/${lm_name}.pocolm

if [ $stage -le 1 ]; then
  # decide on the vocabulary.
  # Note: you'd use --wordlist if you had a previously determined word-list
  # that you wanted to use.
  # Note: if you have more than one order, use a certain amount of words as the
  # vocab and want to restrict max memory for 'sort',
  echo "$0: training the unpruned LM"
  train_lm.py  --wordlist=${wordlist} --num-splits=10 --warm-start-ratio=20  \
               --limit-unk-history=true \
               --fold-dev-into=train ${bypass_metaparam_optim_opt} \
               --min-counts="${min_counts}" \
               ${dir}/data/text ${order} ${lm_dir}/work ${unpruned_lm_dir}

  get_data_prob.py ${dir}/data/test.txt ${unpruned_lm_dir} 2>&1 | grep -F '[perplexity' | tee ${unpruned_lm_dir}/perplexity_test.log

  get_data_prob.py ${dir}/data/real_dev_set.txt ${unpruned_lm_dir} 2>&1 | grep -F '[perplexity' | tee ${unpruned_lm_dir}/perplexity_real_dev_set.log
fi

if [ $stage -le 2 ]; then
  echo "$0: pruning the LM (to larger size)"
  # Using 5 million n-grams for a big LM for rescoring purposes.
  prune_lm_dir.py --target-num-ngrams=$num_ngrams_large --initial-threshold=0.02 ${unpruned_lm_dir} ${dir}/data/lm_${order}_prune_big
  
  get_data_prob.py ${dir}/data/test.txt ${dir}/data/lm_${order}_prune_big 2>&1 | grep -F '[perplexity' | tee ${dir}/data/lm_${order}_prune_big/perplexity_test.log 

  get_data_prob.py ${dir}/data/real_dev_set.txt ${dir}/data/lm_${order}_prune_big 2>&1 | grep -F '[perplexity' | tee ${dir}/data/lm_${order}_prune_big/perplexity_real_dev_set.log

  mkdir -p ${dir}/data/arpa
  format_arpa_lm.py ${dir}/data/lm_${order}_prune_big | gzip -c > ${dir}/data/arpa/${order}gram_big.arpa.gz
fi

if [ $stage -le 3 ]; then
  echo "$0: pruning the LM (to smaller size)"
  # Using 2.5 million n-grams for a smaller LM for graph building.  
  # Prune from the bigger-pruned LM, it'll be faster.
  prune_lm_dir.py --target-num-ngrams=$num_ngrams_small ${dir}/data/lm_${order}_prune_big ${dir}/data/lm_${order}_prune_small

  get_data_prob.py ${dir}/data/test.txt ${dir}/data/lm_${order}_prune_small 2>&1 | grep -F '[perplexity' | tee ${dir}/data/lm_${order}_prune_small/perplexity_test.log 

  get_data_prob.py ${dir}/data/real_dev_set.txt ${dir}/data/lm_${order}_prune_small 2>&1 | grep -F '[perplexity' | tee ${dir}/data/lm_${order}_prune_small/perplexity_real_dev_set.log

  format_arpa_lm.py ${dir}/data/lm_${order}_prune_small | gzip -c > ${dir}/data/arpa/${order}gram_small.arpa.gz
fi
