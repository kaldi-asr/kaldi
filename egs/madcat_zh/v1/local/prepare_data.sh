#!/bin/bash

# Copyright      2017  Chun Chieh Chang
#                2017  Ashish Arora
#                2017  Hossein Hadian
# Apache 2.0

# This script downloads the IAM handwriting database and prepares the training
# and test data (i.e text, images.scp, utt2spk and spk2utt) by calling process_data.py.
# It also downloads the LOB and Brown text corpora. It downloads the database files
# only if they do not already exist in download directory.

#  Eg. local/prepare_data.sh
#  Eg. text file: 000_a01-000u-00 A MOVE to stop Mr. Gaitskell from
#      utt2spk file: 000_a01-000u-00 000
#      images.scp file: 000_a01-000u-00 data/local/lines/a01/a01-000u/a01-000u-00.png
#      spk2utt file: 000 000_a01-000u-00 000_a01-000u-01 000_a01-000u-02 000_a01-000u-03

stage=0
download_dir=data/download/tmp/LDC2014T13/data
train_split_file=data/download/tmp/madcat_datasplit/zh-en/madcat.train.raw.lineid
test_split_file=data/download/tmp/madcat_datasplit/zh-en/madcat.test.raw.lineid
dev_split_file=data/download/tmp/madcat_datasplit/zh-en/madcat.dev.raw.lineid
word_segmented_chinese_data=data/download/tmp/madcat_datasplit/zh-en/madcat.train.tok.zh

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1;

if [[ ! -d $download_dir ]]; then
  echo "$0: Warning: Couldn't find $download_dir."
  echo ""
fi

mkdir -p data/{train,test,dev}
mkdir -p $download_dir/lines
if [ $stage -le 1 ]; then
  local/create_line_image_from_page_image.py $download_dir $train_split_file || exit 1
  local/create_line_image_from_page_image.py $download_dir $test_split_file || exit 1
  local/create_line_image_from_page_image.py $download_dir $dev_split_file || exit 1

  local/process_data.py $download_dir $train_split_file data/train || exit 1
  local/process_data.py $download_dir $test_split_file data/test || exit 1
  local/process_data.py $download_dir $dev_split_file data/dev || exit 1

  for dataset in train test dev; do
    cp data/$dataset/utt2spk data/$dataset/utt2spk_tmp
    cp data/$dataset/text data/$dataset/text_tmp
    cp data/$dataset/images.scp data/$dataset/images_tmp.scp
    sort data/$dataset/utt2spk_tmp > data/$dataset/utt2spk
    sort data/$dataset/text_tmp > data/$dataset/text
    sort data/$dataset/images_tmp.scp > data/$dataset/images.scp
    rm data/$dataset/utt2spk_tmp data/$dataset/text_tmp data/$dataset/images_tmp.scp
  done

  utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
  utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt
  utils/utt2spk_to_spk2utt.pl data/dev/utt2spk > data/dev/spk2utt

  local/process_segments.py $word_segmented_chinese_data data/train
fi
