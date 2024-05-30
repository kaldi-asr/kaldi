#! /bin/bash
# Copyright 2016  Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

encoding="utf_8"

. ./utils/parse_options.sh

command -v morfessor-train >/dev/null 2>&1 || { echo >&2 "Morfessor seems to either not be installed or not on path."; exit 1; }

if [ $# -ne 2 ]; then
  echo >&2 "Usage: ./local/train_morphs.sh [opts] <path_to_wordcounts> <morphs_directory>" 
  echo >&2 "  --encoding <encoding_type> # lexicon encoding" 
  exit 1
fi

input=$1 # wordcounts.txt
output=$2 # data/local/morphs

echo $output
if [ ! -d $output ]; then
  mkdir -p $output
fi

morfessor-train --encoding=utf_8 --traindata-list -f"-_" -s $output/model.bin $input

echo "Finished training morpheme model on $input and stored in $output/model.bin."
