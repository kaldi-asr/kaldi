#!/bin/bash

# Copyright      2017  Chun Chieh Chang
#                2017  Ashish Arora
#                2017  Hossein Hadian
# Apache 2.0

# This script prepares the training and test data for MADCAT Arabic dataset 
# (i.e text, images.scp, utt2spk and spk2utt). It calls process_data.py.

#  Eg. local/prepare_data.sh
#  Eg. text file: LDC0001_000404_NHR_ARB_20070113.0052_11_LDC0001_00z2 ﻮﺠﻫ ﻮﻌﻘﻟ ﻍﺍﺮﻗ ﺢﺗّﻯ ﺎﻠﻨﺧﺎﻋ
#      utt2spk file: LDC0001_000397_NHR_ARB_20070113.0052_11_LDC0001_00z1 LDC0001
#      images.scp file: LDC0009_000000_arb-NG-2-76513-5612324_2_LDC0009_00z0
#      data/local/lines/1/arb-NG-2-76513-5612324_2_LDC0009_00z0.tif

stage=0
download_dir1=/export/corpora/LDC/LDC2012T15/data
download_dir2=/export/corpora/LDC/LDC2013T09/data
download_dir3=/export/corpora/LDC/LDC2013T15/data
train_split_file=/home/kduh/proj/scale2018/data/madcat_datasplit/ar-en/madcat.train.raw.lineid
test_split_file=/home/kduh/proj/scale2018/data/madcat_datasplit/ar-en/madcat.test.raw.lineid
dev_split_file=/home/kduh/proj/scale2018/data/madcat_datasplit/ar-en/madcat.dev.raw.lineid

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1;

mkdir -p data/{train,test,dev}
if [ $stage -le 1 ]; then
  local/process_data.py $download_dir1 $download_dir2 $download_dir3 $dev_split_file data/dev data/local/dev/images.scp || exit 1
  local/process_data.py $download_dir1 $download_dir2 $download_dir3 $test_split_file data/test data/local/test/images.scp || exit 1
  local/process_data.py $download_dir1 $download_dir2 $download_dir3 $train_split_file data/train data/local/train/images.scp || exit 1

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

fi
