#!/bin/bash

# Copyright 2018 John Morgan
# Apache 2.0.

# Begin configuration 
subs_src=$1
tmpdir=data/local/tmp
download_dir=$(pwd)
datadir=$(pwd)
# End configuration

# download the subs corpus
if [ ! -f $download_dir/subs.txt.gz ]; then
    wget -O $download_dir/subs.txt.gz $subs_src
else
  echo "$0: The corpus $subs_src was already downloaded."
fi

if [ ! -f $datadir/subs.txt ]; then
  (
    cd $datadir
    zcat < ./subs.txt.gz > subs.txt
  )
  else
    echo "$0: subs file already extracted."
fi
