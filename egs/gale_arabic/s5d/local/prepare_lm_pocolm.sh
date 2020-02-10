#!/usr/bin/env bash

# Dongji Gao

set -e 
set -o pipefail
set -u

stage=0

dir=data/local/pocolm
cmd=run.pl
order=4
extra_text=""

. ./utils/parse_options.sh

lm_dir=${dir}/data
lm_name=10m_${order}

mkdir -p $dir
. ./path.sh
export PATH=$KALDI_ROOT/tools/pocolm/scripts:$PATH

if [ $stage -le 0 ]; then
  mkdir -p ${dir}/data
  mkdir -p ${dir}/data/text

  echo "$0: Getting the data sources"

  rm ${dir}/data/text/* 2>/dev/null || true

  cat data/dev/text | cut -d ' ' -f2- > ${dir}/data/text/dev.txt
  cat data/train/text | cut -d ' ' -f2- > ${dir}/data/text/train.txt
  [ ! -z $extra_text ] &&  [ -f $extra_text ]  && cp $extra_text ${dir}/data/text/giga.txt
#  cp temp/text.2000k ${dir}/data/text/arb_giga_2000k.txt
fi

if [ $stage -le 1 ]; then
  mkdir -p ${dir}/data/work
  if [ ! -f ${dir}/data/work/word_counts/.done ]; then
    get_word_counts.py ${dir}/data/text ${dir}/data/work/word_counts
    touch ${dir}/data/work/word_counts/.done
  fi
fi

lexicon=data/local/dict/lexicon.txt
[ ! -f $lexicon ] && echo "$0: No such file $lexicon" && exit 1;

wordlist=${dir}/data/work/wordlist
if [ $stage -le 2 ]; then
  cut -d ' ' -f1 $lexicon > $wordlist
  wordlist_to_vocab.py --unk-symbol="<UNK>" $wordlist > ${dir}/data/work/vocab_wordlist.txt
  touch ${dir}/data/work/.vocab_wordlist.txt.done
fi

unpruned_lm_dir=${lm_dir}/${lm_name}.pocolm
echo "$unpruned_lm_dir"

if [ $stage -le 3 ]; then
  echo "$0: training the unpruned LM"
  $cmd ${unpruned_lm_dir}/log/train.log \
    train_lm.py --wordlist=$wordlist --num-split=20 --warm-start-ratio=20 \
                --limit-unk-history=false \
                ${dir}/data/text $order ${lm_dir}/work ${unpruned_lm_dir}

  for x in dev; do
    $cmd ${unpruned_lm_dir}/log/compute_data_prob_${x}.log \
      get_data_prob.py ${dir}/data/text/${x}.txt ${unpruned_lm_dir}
    cat ${unpruned_lm_dir}/log/compute_data_prob_${x}.log | grep -F '[perplexity'
  done

  format_arpa_lm.py ${unpruned_lm_dir} | gzip -c > pocolm/lm.$order.gz
fi

if [ $stage -le 4 ]; then
  echo "$0: pruning the LM (to larger size)"
  size=100000000
  prune_lm_dir.py --target-num-ngrams=$size --initial-threshold=0.02 ${unpruned_lm_dir} ${dir}/data/lm_${order}_prune_big

  for x in dev; do
    echo "============ compute perlexity for big lm ================="
    get_data_prob.py ${dir}/data/text/${x}.txt ${dir}/data/lm_${order}_prune_big 2>&1 | grep -F '[perplexity'
  done

  format_arpa_lm.py ${dir}/data/lm_${order}_prune_big | gzip -c > pocolm/lm.prune.${order}.2big.gz
fi
