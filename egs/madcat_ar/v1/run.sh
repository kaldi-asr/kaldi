#!/bin/bash

# Copyright      2017  Chun Chieh Chang
#                2017  Ashish Arora
#                2017  Hossein Hadian

set -e
stage=0
nj=70
decode_gmm=false
# download_dir{1,2,3} points to the database path on the JHU grid. If you have not
# already downloaded the database you can set it to a local directory
# This corpus can be purchased here:
# https://catalog.ldc.upenn.edu/{LDC2012T15,LDC2013T09/,LDC2013T15/}
download_dir1=/export/corpora/LDC/LDC2012T15/data
download_dir2=/export/corpora/LDC/LDC2013T09/data
download_dir3=/export/corpora/LDC/LDC2013T15/data
writing_condition1=/export/corpora/LDC/LDC2012T15/docs/writing_conditions.tab
writing_condition2=/export/corpora/LDC/LDC2013T09/docs/writing_conditions.tab
writing_condition3=/export/corpora/LDC/LDC2013T15/docs/writing_conditions.tab
data_splits_dir=data/download/data_splits
images_scp_dir=data/local
overwrite=false
subset=false
augment=false
use_extra_corpus_text=true
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh
. ./utils/parse_options.sh  # e.g. this parses the above options
                            # if supplied.
./local/check_tools.sh
mkdir -p data/{train,test,dev}/data
mkdir -p data/local/{train,test,dev}

if [ $stage -le 0 ]; then
  if [ -f data/train/text ] && ! $overwrite; then
    echo "$0: Not processing, probably script have run from wrong stage"
    echo "Exiting with status 1 to avoid data corruption"
    exit 1;
  fi
  local/prepare_data.sh --data_splits $data_splits_dir --download_dir1 $download_dir1 \
                         --download_dir2 $download_dir2 --download_dir3 $download_dir3 \
                         --use_extra_corpus_text $use_extra_corpus_text

  for set in test train dev; do
    data_split_file=$data_splits_dir/madcat.$set.raw.lineid
    local/extract_lines.sh --nj $nj --cmd $cmd --data_split_file $data_split_file \
        --download_dir1 $download_dir1 --download_dir2 $download_dir2 \
        --download_dir3 $download_dir3 --writing_condition1 $writing_condition1 \
        --writing_condition2 $writing_condition2 --writing_condition3 $writing_condition3 \
        --data data/local/$set --subset $subset --augment $augment || exit 1
  done

  echo "$0: Processing data..."
  for set in dev train test; do
    local/process_data.py $download_dir1 $download_dir2 $download_dir3 \
      $data_splits_dir/madcat.$set.raw.lineid data/$set $images_scp_dir/$set/images.scp \
      $writing_condition1 $writing_condition2 $writing_condition3 --augment $augment --subset $subset
    image/fix_data_dir.sh data/${set}
  done
fi


if [ $stage -le 1 ]; then
  for dataset in test train; do
    local/extract_features.sh --nj $nj --cmd $cmd --feat-dim 40 data/$dataset
    steps/compute_cmvn_stats.sh data/$dataset || exit 1;
  done
  utils/fix_data_dir.sh data/train
fi

if [ $stage -le 2 ]; then
  echo "$0: Preparing BPE..."
  cut -d' ' -f2- data/train/text | utils/lang/bpe/reverse.py | \
    utils/lang/bpe/prepend_words.py | \
    utils/lang/bpe/learn_bpe.py -s 700 > data/local/bpe.txt

  for set in test train dev; do
    cut -d' ' -f1 data/$set/text > data/$set/ids
    cut -d' ' -f2- data/$set/text | utils/lang/bpe/reverse.py | \
      utils/lang/bpe/prepend_words.py | \
      utils/lang/bpe/apply_bpe.py -c data/local/bpe.txt \
      | sed 's/@@//g' > data/$set/bpe_text

    mv data/$set/text data/$set/text.old
    paste -d' ' data/$set/ids data/$set/bpe_text > data/$set/text
    rm -f data/$set/bpe_text data/$set/ids
  done

  echo "$0:Preparing dictionary and lang..."
  local/prepare_dict.sh
  utils/prepare_lang.sh --num-sil-states 4 --num-nonsil-states 8 --sil-prob 0.0 --position-dependent-phones false \
                        data/local/dict "<sil>" data/lang/temp data/lang
  utils/lang/bpe/add_final_optional_silence.sh --final-sil-prob 0.5 data/lang
fi

if [ $stage -le 3 ]; then
  echo "$0: Estimating a language model for decoding..."
  local/train_lm.sh
  utils/format_lm.sh data/lang data/local/local_lm/data/arpa/6gram_small.arpa.gz \
                     data/local/dict/lexicon.txt data/lang
  utils/build_const_arpa_lm.sh data/local/local_lm/data/arpa/6gram_unpruned.arpa.gz \
                               data/lang data/lang_rescore_6g
fi

if [ $stage -le 4 ]; then
  steps/train_mono.sh --nj $nj --cmd $cmd --totgauss 10000 data/train \
    data/lang exp/mono
fi

if [ $stage -le 5 ] && $decode_gmm; then
  utils/mkgraph.sh --mono data/lang exp/mono exp/mono/graph

  steps/decode.sh --nj $nj --cmd $cmd exp/mono/graph data/test \
    exp/mono/decode_test
fi

if [ $stage -le 6 ]; then
  steps/align_si.sh --nj $nj --cmd $cmd data/train data/lang \
    exp/mono exp/mono_ali

  steps/train_deltas.sh --cmd $cmd 500 20000 data/train data/lang \
    exp/mono_ali exp/tri
fi

if [ $stage -le 7 ] && $decode_gmm; then
  utils/mkgraph.sh data/lang exp/tri exp/tri/graph

  steps/decode.sh --nj $nj --cmd $cmd exp/tri/graph data/test \
    exp/tri/decode_test
fi

if [ $stage -le 8 ]; then
  steps/align_si.sh --nj $nj --cmd $cmd data/train data/lang \
    exp/tri exp/tri_ali

  steps/train_lda_mllt.sh --cmd $cmd \
    --splice-opts "--left-context=3 --right-context=3" 500 20000 \
    data/train data/lang exp/tri_ali exp/tri3
fi

if [ $stage -le 9 ] && $decode_gmm; then
  utils/mkgraph.sh data/lang exp/tri3 exp/tri3/graph

  steps/decode.sh --nj $nj --cmd $cmd exp/tri3/graph \
    data/test exp/tri3/decode_test
fi

if [ $stage -le 10 ]; then
  steps/align_fmllr.sh --nj $nj --cmd $cmd --use-graphs true \
    data/train data/lang exp/tri3 exp/tri3_ali
fi

if [ $stage -le 11 ]; then
  local/chain/run_cnn.sh
fi

if [ $stage -le 12 ]; then
  local/chain/run_cnn_chainali.sh --stage 2
fi
