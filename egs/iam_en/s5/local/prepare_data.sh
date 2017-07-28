#!/bin/bash

# copyright 2017 Johns Hopkins University (Ashish Arora)
# Apache 2.0

# This script loads the IAM handwritten dataset

stage=0
nj=20

. ./cmd.sh
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

#download dir
dl_dir=data/download
lines=$dl_dir/lines
xml=$dl_dir/xml
ascii=$dl_dir/ascii
dataSplitInfo=$dl_dir/largeWriterIndependentTextLineRecognitionTask
lines_url=http://www.fki.inf.unibe.ch/DBs/iamDB/data/lines/lines.tgz
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

mkdir -p data/{train,val_1,val_2,test}
if [ $stage -le 0 ]; then
  local/process_data.py $dl_dir data/train --dataset trainset --model_type word || exit 1
  local/process_data.py $dl_dir data/val_1 --dataset validationset1 --model_type word || exit 1
  local/process_data.py $dl_dir data/val_2 --dataset validationset2 --model_type word || exit 1
  local/process_data.py $dl_dir data/test --dataset testset --model_type word || exit 1

  utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
  utils/utt2spk_to_spk2utt.pl data/val_1/utt2spk > data/val_1/spk2utt
  utils/utt2spk_to_spk2utt.pl data/val_2/utt2spk > data/val_2/spk2utt
  utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt

fi

mkdir -p char_data/{train,val_1,val_2,test}
if [ $stage -le 1 ]; then
  local/process_data.py $dl_dir char_data/train --dataset trainset --model_type character || exit 1
  local/process_data.py $dl_dir char_data/val_1 --dataset validationset1 --model_type character || exit 1
  local/process_data.py $dl_dir char_data/val_2 --dataset validationset2 --model_type character || exit 1
  local/process_data.py $dl_dir char_data/test --dataset testset --model_type character || exit 1

  utils/utt2spk_to_spk2utt.pl char_data/train/utt2spk > char_data/train/spk2utt
  utils/utt2spk_to_spk2utt.pl char_data/val_1/utt2spk > char_data/val_1/spk2utt
  utils/utt2spk_to_spk2utt.pl char_data/val_2/utt2spk > char_data/val_2/spk2utt
  utils/utt2spk_to_spk2utt.pl char_data/test/utt2spk > char_data/test/spk2utt

fi
