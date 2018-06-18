#!/bin/bash

# The corpus and lexicon are on openslr.org
lexicon="http://www.openslr.org/resources/34/santiago.tar.gz"
speech="http://www.openslr.org/resources/39/LDC2006S37.tar.gz"

tmpdir=data/local/tmp

# where to put the downloaded speech corpus
download_dir=$tmpdir/speech
data_dir=$download_dir/LDC2006S37/data

mkdir -p $tmpdir/heroico $tmpdir/usma $download_dir

# download the corpus from openslr
if [ ! -f $download_dir/LDC2006S37.tar.gz ]; then
  wget -O $download_dir/LDC2006S37.tar.gz $speech

  (
    cd $download_dir
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
