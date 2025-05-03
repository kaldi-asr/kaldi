#!/bin/bash

# Copyright 2018 John Morgan
# Apache 2.0.

# configuration variables
lex=$@
tmpdir=data/local/tmp
# where to put the downloaded speech corpus
downloaddir=$(pwd)
# Where to put the uncompressed file
datadir=$(pwd)
# end of configuration variable settings

# download the corpus 
if [ ! -f $downloaddir/fr.dict ]; then
  wget -O $downloaddir/fr.dict "$lex"
  (
    cd $downloaddir
  )
else
  echo "$0: The corpus $lex was already downloaded."
fi
