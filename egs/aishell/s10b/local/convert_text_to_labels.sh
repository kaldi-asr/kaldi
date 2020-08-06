#!/bin/bash

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

set -e

echo "$0 $@"  # Print the command line for logging

if [ $# != 2 ]; then
  echo "usage: $0 <data-dir> <lang-dir>"
  exit 1
fi

dir=$1
lang=$2

[[ ! -f $dir/text ]] && echo "file $dir/text does not exist!" && exit 1

for f in lexicon.txt tokens.txt; do
  if [[ ! -f $lang/$f ]]; then
    echo "file $lang/$f does not exist!"
    exit 1
  fi
done

python3 ./local/convert_text_to_labels.py \
  --lexicon-filename $lang/lexicon.txt \
  --tokens-filename $lang/tokens.txt \
  --dir $dir
