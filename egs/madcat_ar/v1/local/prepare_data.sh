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
data_splits=data/download/data_splits

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1;

mkdir -p data/{train,test,dev}

if [ $stage -le 1 ]; then
  echo "$0: Processing dev, train and test data..."
  echo "Date: $(date)."
  local/process_data.py $download_dir1 $download_dir2 $download_dir3 $data_splits/madcat.dev.raw.lineid data/dev data/local/dev/images.scp || exit 1
  local/process_data.py $download_dir1 $download_dir2 $download_dir3 $data_splits/madcat.test.raw.lineid data/test data/local/test/images.scp || exit 1
  local/process_data.py $download_dir1 $download_dir2 $download_dir3 $data_splits/madcat.train.raw.lineid data/train data/local/train/images.scp || exit 1

  for dataset in dev test train; do
    echo "$0: Fixing data directory for dataset: $dataset"
    echo "Date: $(date)."
    image/fix_data_dir.sh data/$dataset
  done
fi
