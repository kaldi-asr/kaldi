#!/bin/bash

# Copyright 2018 John Morgan
# Apache 2.0.

# The corpus and lexicon are on openslr.org
lexicon="http://www.openslr.org/resources/34/santiago.tar.gz"
speech="http://www.openslr.org/resources/39/LDC2006S37.tar.gz"

download_dir=$(pwd)
tmpdir=data/local/tmp
data_dir=$tmpdir/LDC2006S37/data

mkdir -p $tmpdir

# download the corpus from openslr
if [ ! -f $download_dir/LDC2006S37.tar.gz ]; then
  wget -O $download_dir/heroico.tar.gz $speech

  (
    cd $download_dir
    tar -xzf heroico.tar.gz
  )
fi

mkdir -p data/local/dict $tmpdir/dict

# download the dictionary from openslr
if [ ! -f $download_dir/santiago.tar.gz ]; then
    wget -O $download_dir/santiago.tar.gz $lexicon
fi

(
  cd $download_dir
  tar -xzf santiago.tar.gz
)
