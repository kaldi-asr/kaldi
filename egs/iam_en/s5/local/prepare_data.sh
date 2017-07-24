#!/bin/bash

# copyright 2017 Johns Hopkins University (Ashish Arora)
# Apache 2.0

# This script loads the IAM handwritten dataset

stage=0

. ./cmd.sh
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

#download dir
dl_dir=data/download
lines=$dl_dir/lines
xml=$dl_dir/xml
lines_url=http://www.fki.inf.unibe.ch/DBs/iamDB/data/lines/lines.tgz
xml_url=http://www.fki.inf.unibe.ch/DBs/iamDB/data/xml/xml.tgz


mkdir -p $dl_dir
#download and extact images and transcription
if [ -d $lines ]; then
  echo Not downloading lines images as it is already there.
else
  if [ ! -f $dl_dir/lines.tgz ]; then
    echo Downloading lines images...
    wget -P $dl_dir --user userjh --password password $lines_url || exit 1;
  fi
  mkdir -p $lines
  tar -xvzf $dl_dir/lines.tgz -C $lines || exit 1;
  echo Done downloading and extracting lines images
fi

if [ -d $xml ]; then
  echo Not downloading transcription as it is already there.
else
  if [ ! -f $dl_dir/xml.tgz ]; then
    echo Downloading transcription ...
    wget -P $dl_dir --user userjh --password password $xml_url || exit 1;
  fi
  mkdir -p $xml
  tar -xvzf $dl_dir/xml.tgz -C $xml || exit 1;
  echo Done downloading and extracting transcription
fi

if [ $stage -le 0 ]; then
  local/process_data.py $dl_dir data/train || exit 1
fi


numsplit=5
mkdir -p data/{train,val_1,val_2,test}/data

if [ $stage -le 1 ]; then
  local/process_feature_vect.py data/train --scale-size 40 | \
  copy-feats --compress=true --compression-method=7 \
  ark:- ark,scp:data/train/data/images.ark,data/train/feats.scp || exit 1
fi

if [ $stage -le 2 ]; then
  mkdir -p data/train/log
  image/split_ocr_dir.sh data/train/ $numsplit
  $cmd JOB=1:$numsplit data/train/log/make_feature_vect.JOB.log \
    local/process_feature_vect.py data/train/split${numsplit}/JOB/ --scale-size 40 \| \
    copy-feats --compress=true --compression-method=7 \
    ark:- ark,scp:data/train/split${numsplit}/JOB/data/images.ark,data/train/split${numsplit}/JOB/feats.scp \
    || exit 1
fi
