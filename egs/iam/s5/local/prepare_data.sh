#!/bin/bash

# copyright 2017 Johns Hopkins University (Ashish Arora)
# Apache 2.0

# This script loads the IAM handwritten dataset

stage=0
nj=20
dir=data

. ./cmd.sh
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

#download dir
dl_dir=data/download
lines=$dl_dir/lines
#lines=$dl_dir/words
xml=$dl_dir/xml
ascii=$dl_dir/ascii
dataSplitInfo=$dl_dir/largeWriterIndependentTextLineRecognitionTask
lines_url=http://www.fki.inf.unibe.ch/DBs/iamDB/data/lines/lines.tgz
#lines_url=http://www.fki.inf.unibe.ch/DBs/iamDB/data/words/words.tgz
xml_url=http://www.fki.inf.unibe.ch/DBs/iamDB/data/xml/xml.tgz
dataSplitInfo_url=http://www.fki.inf.unibe.ch/DBs/iamDB/tasks/largeWriterIndependentTextLineRecognitionTask.zip
ascii_url=http://www.fki.inf.unibe.ch/DBs/iamDB/data/ascii/ascii.tgz
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

if [ -d $ascii ]; then
  echo Not downloading ascii folder as it is already there.
else
  if [ ! -f $dl_dir/ascii.tgz ]; then
    echo Downloading ascii folder ...
    wget -P $dl_dir --user userjh --password password $ascii_url || exit 1;
  fi
  mkdir -p $ascii
  tar -xvzf $dl_dir/ascii.tgz -C $ascii || exit 1;
  echo Done downloading and extracting ascii folder
fi

mkdir -p $dir/{train,val_1,val_2,test}
if [ $stage -le 0 ]; then
  local/process_data.py $dl_dir $dir/train --dataset trainset --model_type word || exit 1
  local/process_data.py $dl_dir $dir/val_1 --dataset validationset1 --model_type word || exit 1
  local/process_data.py $dl_dir $dir/val_2 --dataset validationset2 --model_type word || exit 1
  local/process_data.py $dl_dir $dir/test --dataset testset --model_type word || exit 1

  utils/utt2spk_to_spk2utt.pl $dir/train/utt2spk > $dir/train/spk2utt
  utils/utt2spk_to_spk2utt.pl $dir/val_1/utt2spk > $dir/val_1/spk2utt
  utils/utt2spk_to_spk2utt.pl $dir/val_2/utt2spk > $dir/val_2/spk2utt
  utils/utt2spk_to_spk2utt.pl $dir/test/utt2spk > $dir/test/spk2utt
fi
