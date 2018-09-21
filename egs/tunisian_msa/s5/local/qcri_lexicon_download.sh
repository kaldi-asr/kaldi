#!/bin/bash 

# Copyright 2018 John Morgan
# Apache 2.0.

# configuration variables
lex=$1
tmpdir=data/local/tmp
# where to put the downloaded speech corpus
downloaddir=$(pwd)
# Where to put the uncompressed file
datadir=$(pwd)
# end of configuration variable settings

# download the corpus 
if [ ! -f $downloaddir/qcri.txt.bz2 ]; then
  wget -O $downloaddir/qcri.txt.bz2 $lex
  (
    cd $downloaddir
    bzcat qcri.txt.bz2 | tail -n+4 > $datadir/qcri.txt
  )
else
  echo "$0: The corpus $lex was already downloaded."
fi
