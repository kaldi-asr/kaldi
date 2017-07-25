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
dataSplitInfo=$dl_dir/largeWriterIndependentTextLineRecognitionTask
lines_url=http://www.fki.inf.unibe.ch/DBs/iamDB/data/lines/lines.tgz
xml_url=http://www.fki.inf.unibe.ch/DBs/iamDB/data/xml/xml.tgz
dataSplitInfo_url=http://www.fki.inf.unibe.ch/DBs/iamDB/tasks/largeWriterIndependentTextLineRecognitionTask.zip

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

if [ -d $dataSplitInfo ]; then
  echo Not downloading data split, training and testing split, information as it is already there.
else
  if [ ! -f $dl_dir/largeWriterIndependentTextLineRecognitionTask.zip ]; then
    echo Downloading training and testing data Split Information ...
    wget -P $dl_dir --user userjh --password password $dataSplitInfo_url || exit 1;
  fi
  mkdir -p $dataSplitInfo
  unzip $dl_dir/largeWriterIndependentTextLineRecognitionTask.zip -d $dataSplitInfo || exit 1;
  echo Done downloading and extracting training and testing data Split Information
fi

mkdir -p data/{train,val_1,val_2,test}
if [ $stage -le 0 ]; then
  local/process_data.py $dl_dir data/train --dataset trainset || exit 1
  local/process_data.py $dl_dir data/val_1 --dataset validationset1 || exit 1
  local/process_data.py $dl_dir data/val_2 --dataset validationset2 || exit 1
  local/process_data.py $dl_dir data/test --dataset testset || exit 1

  utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
  utils/utt2spk_to_spk2utt.pl data/val_1/utt2spk > data/val_1/spk2utt
  utils/utt2spk_to_spk2utt.pl data/val_2/utt2spk > data/val_2/spk2utt
  utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt
fi

numsplit=5
mkdir -p data/{train,val_1,val_2,test}/data

if [ $stage -le 1 ]; then
  local/process_feature_vect.py data/train --scale-size 40 | \
  copy-feats --compress=true --compression-method=7 \
  ark:- ark,scp:data/train/data/images.ark,data/train/feats.scp || exit 1

  local/process_feature_vect.py data/val_1 --scale-size 40 | \
  copy-feats --compress=true --compression-method=7 \
  ark:- ark,scp:data/val_1/data/images.ark,data/val_1/feats.scp || exit 1

  local/process_feature_vect.py data/val_2 --scale-size 40 | \
  copy-feats --compress=true --compression-method=7 \
  ark:- ark,scp:data/val_2/data/images.ark,data/val_2/feats.scp || exit 1

  local/process_feature_vect.py data/test --scale-size 40 | \
  copy-feats --compress=true --compression-method=7 \
  ark:- ark,scp:data/test/data/images.ark,data/test/feats.scp || exit 1
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
