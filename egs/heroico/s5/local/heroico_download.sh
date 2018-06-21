#!/bin/bash

# Copyright 2018 John Morgan
# Apache 2.0.

# The corpus and lexicon are on openslr.org
lexicon="http://www.openslr.org/resources/34/santiago.tar.gz"
speech="http://www.openslr.org/resources/39/LDC2006S37.tar.gz"

tmpdir=data/local/tmp
data_dir=$tmpdir/LDC2006S37/data

mkdir -p $tmpdir

# download the corpus from openslr
if [ ! -f $tmpdir/LDC2006S37.tar.gz ]; then
  wget -O $tmpdir/LDC2006S37.tar.gz $speech

  (
    cd $tmpdir
    tar -xzf LDC2006S37.tar.gz
  )
fi

mkdir -p data/local/dict $tmpdir/dict

# download the dictionary from openslr
if [ ! -f $tmpdir/dict/santiago.tar.gz ]; then
    wget -O $tmpdir/dict/santiago.tar.gz $lexicon
fi

(
  cd $tmpdir/dict
  tar -xzf santiago.tar.gz
)
