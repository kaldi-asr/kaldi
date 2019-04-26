#!/bin/bash

# Copyright      2018  Desh Raj (Johns Hopkins University) 

# Apache 2.0

# This script downloads the Bentham handwriting database and prepares the training
# and test data (i.e text, images.scp, utt2spk and spk2utt) by calling create_splits.sh.

# In addition, it downloads data for all texts of Bentham for LM training purpose.

stage=0
download_dir=data/local/download/
database_dir=""
text_corpus_dir=""

mkdir -p $download_dir

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1;

BENTHAM_IMAGES_URL='http://transcriptorium.eu/~tsdata/BenthamR0/BenthamDatasetR0-Images.zip'
BENTHAM_GT_URL='http://transcriptorium.eu/~tsdata/BenthamR0/BenthamDatasetR0-GT.zip'
bentham_images=$database_dir"/images.zip"
bentham_gt=$database_dir"/gt.zip"
bentham_text=$download_dir"/text"

# download and extract images and transcriptions
if [ ! -f $bentham_images ]; then
  echo "Downloading images and transcriptions to $database_dir"
  mkdir -p $database_dir
  wget $BENTHAM_IMAGES_URL -O $bentham_images
  wget $BENTHAM_GT_URL -O $bentham_gt
else
  echo "Not downloading since corpus already exists"
fi

if [ ! -d $download_dir/"gt" ]; then
  unzip $bentham_gt -d $download_dir
  mv $download_dir"/BenthamDatasetR0-GT" $download_dir"/gt"
else
  echo "Local extracted corpus already exists"
fi

# Download extra Bentham text for LM training
if [ -d $text_corpus_dir ]; then
  echo "$0: Not downloading Bentham text corpus as it is already there."
else
  local/download_bentham_text.sh $text_corpus_dir
fi

# Copy extra Bentham text to local
if [ -d $bentham_text ]; then
  echo "$0: Not copying as local Bentham already present."
else
  mkdir -p $bentham_text
  cp $text_corpus_dir/Bentham-Text/* $bentham_text
  echo "$0: Done copying extra Bentham text to local."
fi

# Creating train, val, and test splits for all directories
if [ -d data/train ]; then
  echo "Data splits and files already exist. Not creating again."
else
  echo "Creating train, val, and test splits and corresponding files.."
  local/create_splits.sh $download_dir "data/"
fi

