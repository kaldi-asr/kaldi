#!/bin/bash

# Copyright 2018 John Morgan
# Apache 2.0.

speech=$1

# where to put the downloaded speech corpus
download_dir=$(pwd)
data_dir=$download_dir/Tunisian_MSA/data

# download the corpus from openslr
if [ ! -f $download_dir/tamsa.tar.gz ]; then
    wget -O $download_dir/tamsa.tar.gz $speech
else
  echo "$0: The corpus $speech was already downloaded."
fi

if [ ! -d $download_dir/Tunisian_MSA ]; then
  (
    cd $download_dir
    tar -xzf tamsa.tar.gz
  )
else
  echo "$0: The corpus was already unzipped."
fi
