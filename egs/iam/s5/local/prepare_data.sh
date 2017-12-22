#!/bin/bash

#Copyright      2017  Chun Chieh Chang
#               2017  Ashish Arora

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
download_dir=data/download

. ./cmd.sh
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

lines=data/local/lines
xml=data/local/xml
ascii=data/local/ascii
bcorpus=data/local/browncorpus
lobcorpus=data/local/lobcorpus
data_split_info=data/local/largeWriterIndependentTextLineRecognitionTask
lines_url=http://www.fki.inf.unibe.ch/DBs/iamDB/data/lines/lines.tgz
xml_url=http://www.fki.inf.unibe.ch/DBs/iamDB/data/xml/xml.tgz
data_split_info_url=http://www.fki.inf.unibe.ch/DBs/iamDB/tasks/largeWriterIndependentTextLineRecognitionTask.zip
ascii_url=http://www.fki.inf.unibe.ch/DBs/iamDB/data/ascii/ascii.tgz
brown_corpus_url=http://www.sls.hawaii.edu/bley-vroman/brown.txt
lob_corpus_url=http://ota.ox.ac.uk/text/0167.zip
mkdir -p $download_dir data/local

# download and extact images and transcription
if [ -d $lines ]; then
  echo Not downloading lines images as it is already there.
else
  if [ ! -f $download_dir/lines.tgz ]; then
    echo Downloading lines images...
    wget -P $download_dir --user userjh --password password $lines_url || exit 1;
  fi
  mkdir -p $lines
  tar -xzf $download_dir/lines.tgz -C $lines || exit 1;
  echo Done downloading and extracting lines images
fi

if [ -d $xml ]; then
  echo Not downloading transcription as it is already there.
else
  if [ ! -f $download_dir/xml.tgz ]; then
    echo Downloading transcription ...
    wget -P $download_dir --user userjh --password password $xml_url || exit 1;
  fi
  mkdir -p $xml
  tar -xzf $download_dir/xml.tgz -C $xml || exit 1;
  echo Done downloading and extracting transcription
fi

if [ -d $data_split_info ]; then
  echo Not downloading data split, training and testing split, information as it is already there.
else
  if [ ! -f $download_dir/largeWriterIndependentTextLineRecognitionTask.zip ]; then
    echo Downloading training and testing data Split Information ...
    wget -P $download_dir --user userjh --password password $data_split_info_url || exit 1;
  fi
  mkdir -p $data_split_info
  unzip $download_dir/largeWriterIndependentTextLineRecognitionTask.zip -d $data_split_info || exit 1;
  echo Done downloading and extracting training and testing data Split Information
fi

if [ -d $ascii ]; then
  echo Not downloading ascii folder as it is already there.
else
  if [ ! -f $download_dir/ascii.tgz ]; then
    echo Downloading ascii folder ...
    wget -P $download_dir --user userjh --password password $ascii_url || exit 1;
  fi
  mkdir -p $ascii
  tar -xzf $download_dir/ascii.tgz -C $ascii || exit 1;
  echo Done downloading and extracting ascii folder
fi

if [ -d $lobcorpus ]; then
  echo Not downloading lob corpus as it is already there.
else
  if [ ! -f $download_dir/0167.zip ]; then
    echo Downloading lob corpus ...
    wget -P $download_dir $lob_corpus_url || exit 1;
  fi
  mkdir -p $lobcorpus
  unzip $download_dir/0167.zip -d $lobcorpus || exit 1;
  echo Done downloading and extracting lob corpus
fi

if [ -d $bcorpus ]; then
  echo Not downloading brown corpus as it is already there.
else
  if [ ! -f $bcorpus/brown.txt ]; then
    mkdir -p $bcorpus
    echo Downloading brown corpus ...
    wget -P $bcorpus $brown_corpus_url || exit 1;
  fi
  echo Done downloading brown corpus
fi

mkdir -p data/{train,test,val}
file_name=largeWriterIndependentTextLineRecognitionTask

train_old="data/local/$file_name/trainset.txt"
test_old="data/local/$file_name/testset.txt"
val1_old="data/local/$file_name/validationset1.txt"
val2_old="data/local/$file_name/validationset2.txt"

train_new="data/local/new_trainset.txt"
test_new="data/local/new_testset.txt"
val_new="data/local/new_valset.txt"

cat $train_old > $train_new
cat $test_old > $test_new
cat $val1_old $val2_old > $val_new

if [ $stage -le 0 ]; then
  local/process_data.py data/local data/train --dataset new_trainset || exit 1
  local/process_data.py data/local data/test --dataset new_testset || exit 1
  local/process_data.py data/local data/val --dataset new_valset || exit 1

  utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
  utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt
fi
