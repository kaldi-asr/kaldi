#!/bin/bash

# Copyright 2014 Vassil Panayotov
# Apache 2.0

# Performs text normalization for subsequent language model training

echo $@

. path.sh || exit 1

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <input-book-dirs> <output-root>"
  exit 1
fi

in_list=$1
out_root=$2

[[ -f "$in_list" ]] || { echo "The input file '$in_list' does not exists!"; exit 1; }

command -v nsw_expand 1>/dev/null 2>&1 || {
  echo ""
  echo "The Festival's NSW text normalization package is not found in PATH";
  echo "You can try to install it by running:";
  echo "  local/lm/install_festival.sh [--apply-gcc-patch false]";
  echo "Note however, that this script should only be considered as an example,";
  echo "so if you run into installation problems, it's up to you to resolve them";
  exit 1;
}

mkdir -p $out_root

processed=0
for b in $(cat $in_list); do
  id=$(basename $b)
  echo "Start processing $id at $(date '+%T %F')"
  in_file=$b/$id.txt
  [[ -f "$in_file" ]] || { echo "WARNING: $in_file does not exists"; continue; }
  out_file=$out_root/$id/$id.txt
  mkdir -p $out_root/$id
  $PYTHON local/lm/python/pre_filter.py $in_file /dev/stdout |\
    $PYTHON local/lm/python/text_pre_process.py /dev/stdin /dev/stdout |\
    nsw_expand -format opl /dev/stdin |\
    $PYTHON local/lm/python/text_post_process.py /dev/stdin $out_file /dev/null || exit 1
  processed=$((processed + 1))
  echo "Processing of $id has finished at $(date '+%T %F') [$processed texts ready so far]"
done

echo "$processed texts processed OK and stored under '$out_root'"

exit 0
