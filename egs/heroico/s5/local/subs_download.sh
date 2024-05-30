#!/usr/bin/env bash

# Copyright 2017 John Morgan
# Apache 2.0.

tmpdir=data/local/tmp
download_dir=$(pwd)
mkdir -p $download_dir
subs_src=$1

# download the subs corpus
if [ ! -f $download_dir/subs.zip ]; then
  wget -O $download_dir/subs.zip $subs_src
  (
    cd $download_dir
    unzip subs.zip
  )
  else
    echo "$0: subs file already downloaded."
fi
