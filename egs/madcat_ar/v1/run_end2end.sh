#!/bin/bash
# Copyright 2017    Hossein Hadian
#           2018    Ashish Arora
set -e
stage=0
nj=70
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
  for dataset in test dev train; do
    echo "$0: Extracting line images from page image for dataset:  $dataset. "
    echo "Date: $(date)."
    dataset_file=$data_splits_dir/madcat.$dataset.raw.lineid
    local/extract_lines.sh --nj $nj --cmd $cmd --dataset_file $dataset_file \
                           --download_dir1 $download_dir1 --download_dir2 $download_dir2 \
                           --download_dir3 $download_dir3 data/local/$dataset
  done
fi

if [ $stage -le 2 ]; then
  echo "$0: Preparing dev train and eval data..."
  echo "Date: $(date)."
  local/prepare_data.sh
fi

if [ $stage -le 3 ]; then
  echo "$0: Obtaining image groups. calling get_image2num_frames"
  echo "Date: $(date)."
  image/get_image2num_frames.py data/train  # This will be needed for the next command
  # The next command creates a "allowed_lengths.txt" file in data/train
  # which will be used by local/make_features.py to enforce the images to
  # have allowed lengths. The allowed lengths will be spaced by 10% difference in length.
  echo "$0: Obtaining image groups. calling get_allowed_lengths"
  echo "Date: $(date)."
  image/get_allowed_lengths.py --frame-subsampling-factor 4 10 data/train
fi

if [ $stage -le 4 ]; then
  for dataset in test train; do
    echo "$0: Extracting features and calling compute_cmvn_stats for dataset:  $dataset. "
    echo "Date: $(date)."
    local/extract_features.sh --nj $nj --cmd $cmd --feat-dim 40 data/$dataset
    steps/compute_cmvn_stats.sh data/$dataset || exit 1;
  done
  echo "$0: Fixing data directory for train dataset"
  echo "Date: $(date)."
  utils/fix_data_dir.sh data/train
fi

if [ $stage -le 5 ]; then
  echo "$0: Preparing dictionary and lang..."
  local/prepare_dict.sh
  utils/prepare_lang.sh --num-sil-states 4 --num-nonsil-states 8 --sil-prob 0.9999 \
                        data/local/dict "<sil>" data/lang/temp data/lang
fi

if [ $stage -le 6 ]; then
  echo "$0: Estimating a language model for decoding..."
  local/train_lm.sh
  utils/format_lm.sh data/lang data/local/local_lm/data/arpa/3gram_unpruned.arpa.gz \
                     data/local/dict/lexicon.txt data/lang_test
fi

if [ $stage -le 7 ]; then
  echo "$0: Calling the flat-start chain recipe..."
  echo "Date: $(date)."
  local/chain/run_flatstart_cnn1a.sh --nj $nj
fi

if [ $stage -le 8 ]; then
  echo "$0: Aligning the training data using the e2e chain model..."
  echo "Date: $(date)."
  steps/nnet3/align.sh --nj $nj --cmd "$cmd" \
                       --scale-opts '--transition-scale=1.0 --self-loop-scale=1.0' \
                       data/train data/lang exp/chain/e2e_cnn_1a exp/chain/e2e_ali_train
fi

if [ $stage -le 9 ]; then
  echo "$0: Building a tree and training a regular chain model using the e2e alignments..."
  echo "Date: $(date)."
  local/chain/run_cnn_e2eali_1b.sh --nj $nj
fi
