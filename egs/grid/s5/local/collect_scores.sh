#!/bin/bash

# Copyright 2017  Ruhr-University Bochum (Author: Hendrik Meutzner)
#           
# Apache 2.0.
#
# This script collects and prints all scores that are found in the exp directory.
# If no arguments are provided, the exp directory is assumed to be located in pwd.
# Usage: collect_scores.sh [exp-dir]

if [ $# != 1 ]; then
  expdir=exp
else
  expdir=$1
fi

flist=`ls ${expdir}/*/*/decode*/keyword_scores.txt`

for file in $flist; do
  echo $file | sed -e 's|'$expdir'||g' | cut -d '.' -f 1; cat $file;
  echo ""
done
