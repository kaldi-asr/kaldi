#!/usr/bin/env bash

# Copyright      2017  Chun Chieh Chang
#                2017  Ashish Arora
#                2017  Hossein Hadian

set -e
stage=0
nj=50
decode_gmm=true
# madcat_database points to the database path on the JHU grid. If you have not
# already downloaded the database you can set it to a local directory
# like "data/download" and follow the instructions
# in "local/download_data.sh" to download the database:
# data_split_dir is an unofficial datasplit that is used.
# The datasplits can be found on http://www.openslr.org/51/
madcat_database=/export/corpora/LDC/LDC2014T13
data_split_dir=data/download/datasplits
overwrite=false
corpus_dir=/export/corpora5/handwriting_ocr/corpus_data/zh/

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh
. ./utils/parse_options.sh  # e.g. this parses the above options
                            # if supplied.
./local/check_tools.sh

# Start from stage=-1 for using extra corpus text
if [ $stage -le -1 ]; then
  echo "$(date): getting corpus text for language modelling..."
  mkdir -p data/local/text/cleaned
  cat $corpus_dir/* > data/local/text/zh.txt
  head -20000 data/local/text/zh.txt > data/local/text/cleaned/val.txt
  tail -n +20000 data/local/text/zh.txt > data/local/text/cleaned/corpus.txt
fi

mkdir -p data/{train,test,dev}/lines
if [ $stage -le 0 ]; then

  if [ -f data/train/text ] && ! $overwrite; then
    echo "$0: Not processing, probably script have run from wrong stage"
    echo "Exiting with status 1 to avoid data corruption"
    exit 1;
  fi

   echo "$0: Preparing data..."
  local/prepare_data.sh --download-dir1 $madcat_database/data --data-split-dir $data_split_dir

  for dataset in train test dev; do
    local/extract_lines.sh --nj $nj --cmd $cmd \
      --download-dir $madcat_database
      --dataset-file $data_split_dir/madcat.${dataset}.raw.lineid \
      data/${dataset}/lines
  done

  echo "$0: Processing data..."
  for set in dev train test; do
    local/process_data.py $madcat_database $data_split_dir/madcat.$set.raw.lineid data/$set
    image/fix_data_dir.sh data/$set
  done
fi

mkdir -p data/{train,test,dev}/data
if [ $stage -le 1 ]; then
  for dataset in train test dev; do
    local/extract_features.sh --nj $nj --cmd $cmd --feat-dim 60 data/$dataset
    steps/compute_cmvn_stats.sh data/$dataset
  done
fi

if [ $stage -le 2 ]; then
echo "$0: Preparing dictionary and lang..."
  local/prepare_dict.sh
  utils/prepare_lang.sh --num-sil-states 4 --num-nonsil-states 16 --sil-prob 0.95 \
                        --position-dependent-phones false \
                        data/local/dict "<sil>" data/lang/temp data/lang
fi

if [ $stage -le 3 ]; then
  echo "$0: Estimating a language model for decoding..."
  local/train_lm.sh
  utils/format_lm.sh data/lang data/local/local_lm/data/arpa/3gram_unpruned.arpa.gz \
                     data/local/dict/lexicon.txt data/lang_test
fi

if [ $stage -le 4 ]; then
  steps/train_mono.sh --nj $nj --cmd $cmd --totgauss 10000 data/train \
    data/lang exp/mono
fi

if [ $stage -le 5 ] && $decode_gmm; then
  utils/mkgraph.sh --mono data/lang_test exp/mono exp/mono/graph

  steps/decode.sh --nj $nj --cmd $cmd exp/mono/graph data/test \
    exp/mono/decode_test
fi

if [ $stage -le 6 ]; then
  steps/align_si.sh --nj $nj --cmd $cmd data/train data/lang \
    exp/mono exp/mono_ali

  steps/train_deltas.sh --cmd $cmd --context-opts "--context-width=2 --central-position=1" \
    50000 20000 data/train data/lang \
    exp/mono_ali exp/tri
fi

if [ $stage -le 7 ] && $decode_gmm; then
  utils/mkgraph.sh data/lang_test exp/tri exp/tri/graph

  steps/decode.sh --nj $nj --cmd $cmd exp/tri/graph data/test \
    exp/tri/decode_test
fi

if [ $stage -le 8 ]; then
  steps/align_si.sh --nj $nj --cmd $cmd data/train data/lang \
    exp/tri exp/tri_ali

  steps/train_lda_mllt.sh --cmd $cmd \
    --splice-opts "--left-context=3 --right-context=3" \
    --context-opts "--context-width=2 --central-position=1" 50000 20000 \
    data/train data/lang exp/tri_ali exp/tri2
fi

if [ $stage -le 9 ] && $decode_gmm; then
  utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph

  steps/decode.sh --nj $nj --cmd $cmd exp/tri2/graph \
    data/test exp/tri2/decode_test
fi

if [ $stage -le 10 ]; then
  steps/align_fmllr.sh --nj $nj --cmd $cmd --use-graphs true \
    data/train data/lang exp/tri2 exp/tri2_ali

  steps/train_sat.sh --cmd $cmd --context-opts "--context-width=2 --central-position=1" \
    50000 20000 data/train data/lang \
    exp/tri2_ali exp/tri3
fi

if [ $stage -le 11 ] && $decode_gmm; then
  utils/mkgraph.sh data/lang_test exp/tri3 exp/tri3/graph

  steps/decode_fmllr.sh --nj $nj --cmd $cmd exp/tri3/graph \
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
  local/chain/run_cnn_chainali_1b.sh --chain-model-dir exp/chain/cnn_1a --stage 2
fi
