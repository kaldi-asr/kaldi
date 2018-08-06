#!/bin/bash

# Copyright 2018 John Morgan
# Apache 2.0.

speech=$1
lexicon=$2

download_dir=$(pwd)
tmpdir=data/local/tmp
data_dir=$tmpdir/LDC2006S37/data

mkdir -p $tmpdir

# download the corpus from openslr

if [ ! -f $download_dir/heroico.tar.gz ]; then
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
