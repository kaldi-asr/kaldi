#!/bin/bash
# Copyright 2017    Hossein Hadian

set -e
stage=0
nj=50
username=
password=
# iam_database points to the database path on the JHU grid. If you have not
# already downloaded the database you can set it to a local directory
# like "data/download" and follow the instructions
# in "local/prepare_data.sh" to download the database:
madcat_database=/export/corpora/LDC/LDC2014T13
data_split_dir=data/download/datasplits

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh
. ./utils/parse_options.sh  # e.g. this parses the above options
                            # if supplied.


#./local/check_tools.sh

if [ $stage -le 0 ]; then
  local/download_data.sh --download-dir1 $madcat_database/data --data-split-dir $data_splits_dir

  for dataset in train test dev; do
    local/extract_lines.sh --nj $nj --cmd $cmd \
      --download-dir $madcat_database \
      --dataset-file $data_split_dir/madcat.${dataset}.raw.lineid \
      data/${dataset}/lines
  done

  echo "$0: Preparing data..."
  local/prepare_data.sh --download-dir "$madcat_database"
fi

mkdir -p data/{train,test}/data
if [ $stage -le 1 ]; then
  image/get_image2num_frames.py --feat-dim 80 data/train  # This will be needed for the next command
  
  # The next command creates a "allowed_lengths.txt" file in data/train
  # which will be used by local/make_features.py to enforce the images to
  # have allowed lengths. The allowed lengths will be spaced by 10% difference in length.
  
  image/get_allowed_lengths.py --frame-subsampling-factor 4 10 data/train
  echo "$0: Preparing the test and train feature files..."
  for dataset in train test; do
    local/extract_features.sh --nj $nj --cmd $cmd --feat-dim 80 data/$dataset
    steps/compute_cmvn_stats.sh data/$dataset
  done
  utils/fix_data_dir.sh data/train
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
  echo "$0: calling the flat-start chain recipe..."
  local/chain/run_flatstart_cnn1a.sh
fi
