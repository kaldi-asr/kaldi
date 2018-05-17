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
# https://catalog.ldc.upenn.edu/LDC2012T15,
# https://catalog.ldc.upenn.edu/LDC2013T09/,
# https://catalog.ldc.upenn.edu/LDC2013T15/.
download_dir1=/export/corpora/LDC/LDC2012T15/data
download_dir2=/export/corpora/LDC/LDC2013T09/data
download_dir3=/export/corpora/LDC/LDC2013T15/data
data_splits_dir=data/download/data_splits

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh
. ./utils/parse_options.sh  # e.g. this parses the above options
                            # if supplied.

./local/check_tools.sh

mkdir -p data/{train,test,dev}/data
mkdir -p data/local/{train,test,dev}

if [ $stage -le 0 ]; then
  echo "$0: Downloading data splits..."
  echo "Date: $(date)."
  local/download_data.sh --data_splits $data_splits_dir
fi

if [ $stage -le 1 ]; then
  for dataset in test train dev; do
    dataset_file=$data_splits_dir/madcat.$dataset.raw.lineid
    local/extract_lines.sh --nj $nj --cmd $cmd --dataset_file $dataset_file \
                           --download_dir1 $download_dir1 --download_dir2 $download_dir2 \
                           --download_dir3 $download_dir3 data/local/$dataset
  done
fi

if [ $stage -le 2 ]; then
  echo "$0: Preparing data..."
  local/prepare_data.sh
fi

mkdir -p data/{train,test,dev}/data

if [ $stage -le 3 ]; then
  for dataset in test train; do
    local/extract_features.sh --nj $nj --cmd $cmd --feat-dim 40 data/$dataset
    steps/compute_cmvn_stats.sh data/$dataset || exit 1;
  done
  utils/fix_data_dir.sh data/train
fi

if [ $stage -le 4 ]; then
  echo "$0: Preparing dictionary and lang..."
  local/prepare_dict.sh
  utils/prepare_lang.sh --num-sil-states 4 --num-nonsil-states 8 --sil-prob 0.95 \
    data/local/dict "<sil>" data/lang/temp data/lang
fi

if [ $stage -le 5 ]; then
  echo "$0: Estimating a language model for decoding..."
  local/train_lm.sh
  utils/format_lm.sh data/lang data/local/local_lm/data/arpa/3gram_unpruned.arpa.gz \
                     data/local/dict/lexicon.txt data/lang_test
fi

if [ $stage -le 6 ]; then
  steps/train_mono.sh --nj $nj --cmd $cmd --totgauss 10000 data/train \
    data/lang exp/mono
fi

if [ $stage -le 7 ] && $decode_gmm; then
  utils/mkgraph.sh --mono data/lang_test exp/mono exp/mono/graph

  steps/decode.sh --nj $nj --cmd $cmd exp/mono/graph data/test \
    exp/mono/decode_test
fi

if [ $stage -le 8 ]; then
  steps/align_si.sh --nj $nj --cmd $cmd data/train data/lang \
    exp/mono exp/mono_ali

  steps/train_deltas.sh --cmd $cmd 500 20000 data/train data/lang \
    exp/mono_ali exp/tri
fi

if [ $stage -le 9 ] && $decode_gmm; then
  utils/mkgraph.sh data/lang_test exp/tri exp/tri/graph

  steps/decode.sh --nj $nj --cmd $cmd exp/tri/graph data/test \
    exp/tri/decode_test
fi

if [ $stage -le 10 ]; then
  steps/align_si.sh --nj $nj --cmd $cmd data/train data/lang \
    exp/tri exp/tri_ali

  steps/train_lda_mllt.sh --cmd $cmd \
    --splice-opts "--left-context=3 --right-context=3" 500 20000 \
    data/train data/lang exp/tri_ali exp/tri3
fi

if [ $stage -le 11 ] && $decode_gmm; then
  utils/mkgraph.sh data/lang_test exp/tri3 exp/tri3/graph

  steps/decode.sh --nj $nj --cmd $cmd exp/tri3/graph \
    data/test exp/tri3/decode_test
fi

if [ $stage -le 12 ]; then
  steps/align_fmllr.sh --nj $nj --cmd $cmd --use-graphs true \
    data/train data/lang exp/tri3 exp/tri3_ali
fi

if [ $stage -le 13 ]; then
  local/chain/run_cnn_1a.sh
fi

if [ $stage -le 14 ]; then
  local/chain/run_cnn_chainali_1a.sh --stage 2
fi
