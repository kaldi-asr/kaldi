#!/usr/bin/env bash

# Copyright      2017  Chun Chieh Chang
#                2017  Ashish Arora
#                2017  Hossein Hadian
# Apache 2.0

# This script downloads the data splits for MADCAT Arabic dataset and prepares the training
# validation, and test data (i.e text, images.scp, utt2spk and spk2utt) by calling process_data.py.
# It also uses Arabic Gigaword text corpus for language modeling.

#  Eg. local/prepare_data.sh
#  Eg. text file: LDC0001_000399_NHR_ARB_20070113.0052_11_LDC0001_0z11 
#                 وهناك تداخل بين الرأسمالية الإسرائيلية
#      utt2spk file: LDC0001_000397_NHR_ARB_20070113.0052_11_LDC0001_00z1 LDC0001
#      images.scp file: LDC0001_000397_NHR_ARB_20070113.0052_11_LDC0001_00z1 
#                        data/local/train/1/NHR_ARB_20070113.0052_11_LDC0001_00z1.png

download_dir1=/export/corpora/LDC/LDC2012T15/data
download_dir2=/export/corpora/LDC/LDC2013T09/data
download_dir3=/export/corpora/LDC/LDC2013T15/data
train_split_url=http://www.openslr.org/resources/48/madcat.train.raw.lineid
test_split_url=http://www.openslr.org/resources/48/madcat.test.raw.lineid
dev_split_url=http://www.openslr.org/resources/48/madcat.dev.raw.lineid
data_splits=data/download/data_splits
stage=0
download_dir=data/download
gigacorpus=data/local/gigawordcorpus
gigaword_loc=/export/corpora5/LDC/LDC2011T11
use_extra_corpus_text=true

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1;

if [ -d $data_splits ]; then
  echo "$0: Not downloading the data splits as it is already there."
else
  if [ ! -f $data_splits/madcat.train.raw.lineid ]; then
    mkdir -p $data_splits
    echo "$0: Downloading the data splits..."
    wget -P $data_splits $train_split_url || exit 1;
    wget -P $data_splits $test_split_url || exit 1;
    wget -P $data_splits $dev_split_url || exit 1;
  fi
  echo "$0: Done downloading the data splits"
fi

if [ -d $download_dir1 ]; then
  echo "$0: madcat arabic data directory is present."
else
  if [ ! -f $download_dir1/madcat/*.madcat.xml ]; then
    echo "$0: please download madcat data..."
  fi
fi

mkdir -p $download_dir data/local
if $use_extra_corpus_text; then
  mkdir -p $gigacorpus
  cp -r $gigaword_loc/. $gigacorpus
  for newswire in aaw_arb afp_arb ahr_arb asb_arb hyt_arb nhr_arb qds_arb umh_arb xin_arb; do
    for file in $gigacorpus/arb_gw_5/data/$newswire/*.gz; do
      gzip -d $file
    done
    for file in $gigacorpus/arb_gw_5/data/$newswire/*; do
      sed -e '/^<[^>]*>$/d; s/``/"/g; s/\x27\x27/"/g' $file >> $gigacorpus/arb_gw_5/data/${newswire}_combined.txt
    done
  done
fi
