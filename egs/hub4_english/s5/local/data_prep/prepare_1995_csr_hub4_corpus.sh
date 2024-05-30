#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

# This script prepares the 1995 CSR-IV HUB4 corpus
# https://catalog.ldc.upenn.edu/LDC96S31

set -e
set -o pipefail
set -u

if [ $# -ne 2 ]; then
  echo "Usage: $0 <SOURCE-DIR> <dir>"
  echo " e.g.: $0 /export/corpora5/LDC/LDC96S31/csr95_hub4 data/local/data/csr95_hub4"
  exit 1
fi

SOURCE_DIR=$1
dir=$2

for d in $SOURCE_DIR/csr95/h4/devtst $SOURCE_DIR/csr95/h4/evltst \
  $SOURCE_DIR/csr95/h4/train; do
  if [ ! -d $d ]; then
    echo "$0: Invalid SOURCE-DIR $SOURCE_DIR for LDC96S31 corpus"
    exit 1
  fi
done

mkdir -p $dir

for x in `ls $SOURCE_DIR/csr95/h4/*/*.wav`; do
  y=`basename $x`
  z=${y%.wav}
  echo "$z $x"
done > $dir/wav_scp

cat $dir/wav_scp | grep "csr95/h4/train" > $dir/train95_wav_scp
cat $dir/wav_scp | grep "csr95/h4/devtst" > $dir/dev95_wav_scp
cat $dir/wav_scp | grep "csr95/h4/evltst" > $dir/eval95_wav_scp

rm $dir/*_{segments,utt2spk,text} || true

ls $SOURCE_DIR/csr95/h4/train/*.txt > $dir/train95_text.list
ls $SOURCE_DIR/csr95/h4/devtst/*.txt > $dir/dev95_text.list
ls $SOURCE_DIR/csr95/h4/evltst/*.txt > $dir/eval95_text.list

for x in `ls $SOURCE_DIR/csr95/h4/*/*.txt`; do
  if [[ $x =~ "csr95/h4/train" ]]; then
    local/data_prep/process_1995_bn_annotation.py $x \
      $dir/train95_segments $dir/train95_utt2spk $dir/train95_text
  fi
  
  if [[ $x =~ "csr95/h4/devtst" ]]; then
    local/data_prep/process_1995_bn_annotation.py $x \
      $dir/dev95_segments $dir/dev95_utt2spk $dir/dev95_text
  fi
  
  if [[ $x =~ "csr95/h4/evltst" ]]; then
    local/data_prep/process_1995_bn_annotation.py $x \
      $dir/eval95_segments $dir/eval95_utt2spk $dir/eval95_text
  fi
done
