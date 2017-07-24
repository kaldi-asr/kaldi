#!/bin/bash

# copyright 2017 Johns Hopkins University (Ashish Arora)
# Apache 2.0

# This script loads the IAM handwritten dataset

[ -f ./path.sh ] && . ./path.sh; # source the path.

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

local/process_data.py $dl_dir data/train || exit 1

