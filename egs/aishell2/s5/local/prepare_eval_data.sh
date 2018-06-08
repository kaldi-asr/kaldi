#!/bin/bash
# Copyright 2018 Emotech LTD (Author: Xuechen LIU)
# Apache 2.0

# This script downloads development and evaluation data and prepare necessary files.
# Then it calls local/prepare_data.sh to prepare the data for testing.
# To run this script from scratch, you may need to install unar as below:
# apt install unar

set -e
. ./cmd.sh
. ./path.sh

if [ $# != 2 ]; then
  echo "Usage: $0 <corpus-data-dir> <dict-dir>"
  echo " $0 /export/AISHELL-2/iOS/data data/local/dict "
  exit 1;
fi
 # note: the corpus directory shall be full name
corpus=$1
dict_dir=$2

# check if the downloading and extracting has been done. If not, do it
# note: 'evalT' stores the text files for each recording condition and
# here we only employ iOS condition
if [ ! -d $corpus/DEV ] || [ ! -d $corpus/TEST ] || [ ! -d $corpus/evalT ]; then
  wget https://aishell-eval.oss-cn-beijing.aliyuncs.com/DEV.tar.gz -P $corpus && tar -xvzf $corpus/DEV.tar.gz -C $corpus || exit 1;
  wget https://aishell-eval.oss-cn-beijing.aliyuncs.com/TEST.tar.gz -P $corpus && tar -xvzf $corpus/TEST.tar.gz -C $corpus || exit 1;
  wget https://aishell-eval.oss-cn-beijing.aliyuncs.com/TEST.tar.gz -P $corpus && unar $corpus/evalT.rar $corpus || exit 1;
fi

# prepare wav.scp
for part in DEV TEST; do
  for n in $corpus/$part/IOS/*/*; do printf '%s\n' "$n"; done > $corpus/$part/raw.list || exit 1;
  #IFS=$'\n'       # make newlines the only separator
  for i in $(cat < "$corpus/$part/raw.list"); do
    echo "$i" | rev | cut -d'/' -f 1 | cut -d'.' -f 2 | rev | sort | uniq  >> $corpus/$part/utt.list
    echo "$i" | rev | cut -d'/' -f 1-3 | rev | sort | uniq >> $corpus/$part/wav.list
  done
  paste -d'\t' $corpus/$part/utt.list $corpus/$part/wav.list > $corpus/$part/wav.scp || exit 1;
done

# prepare transcriptions, stored as trans.txt
# according to current architecture, first dev then eval
num_line_dev=$(wc -l $corpus/DEV/wav.scp | awk '{print $1}')
num_line_test=$(wc -l $corpus/TEST/wav.scp | awk '{print $1}')
echo $num_line_dev
head -n $num_line_dev $corpus/evalT/ios.txt | sort -k 1 | uniq > $corpus/DEV/trans.txt || exit 1;
tail -n $num_line_test $corpus/evalT/ios.txt | sort -k 1 | uniq > $corpus/TEST/trans.txt || exit 1;

# then prepare data formally
for part in DEV TEST; do
  lower_part=$(echo $part | tr '[:upper:]' '[:lower:]')
  local/prepare_data.sh $corpus/$part $dict_dir data/$lower_part || exit 1;
done


