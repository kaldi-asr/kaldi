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
download_dir=data/download/tmp
username=
password=       # username and password for downloading the IAM database
                # if you have not already downloaded the database, please
                # register at http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
                # and provide this script with your username and password.

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1;

if [[ ! -d $download_dir ]]; then
  echo "$0: Warning: Couldn't find $download_dir."
  echo ""
fi

mkdir -p data/{train,test,val}
mkdir -p $download_dir/lines
if [ $stage -le 1 ]; then
  local/create_line_image_from_page_image.py $download_dir/LDC2014T13/data

  local/process_data.py $download_dir data/train || exit 1
  local/process_data.py $download_dir data/test || exit 1
  local/process_data.py $download_dir data/val || exit 1

  utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
  utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt
fi
