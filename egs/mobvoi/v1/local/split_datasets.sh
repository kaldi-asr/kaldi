#!/bin/bash
# Copyright 2018-2020  Yiming Wang
#           2018-2020  Daniel Povey
# Apache 2.0

stage=0

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -eu


if [ $stage -le 1 ]; then
  dir=data/download/ticmini2_dataset_20180607
  trans=$dir/hixiaowen.txt
  paste -d ' ' <(cat $trans | awk '{split($1,a,"."); print a[1]}') <(cat $trans | cut -d ' ' -f2-) > $dir/hixiaowen_text || exit 1
  dir=data/download/ticmini2_hixiaowen_adult_20180731
  for folder in patch1 patch2; do
    trans=$dir/$folder/hixiaowen_trans
    paste -d ' ' <(cat $trans | awk '{split($1,a,"."); print a[1]}' | awk '{split($1,a,"/"); print a[3]}') <(cat $trans | cut -d ' ' -f2-) || exit 1
  done > $dir/hixiaowen_text || exit 1
  dir=data/download/ticmini2_for_school_20180911
  trans=$dir/hixiaowen/hixiaowen.trans
  paste -d ' ' <(cat $trans | awk '{split($1,a,"/"); print a[4]}' | awk '{split($1,a,"."); print a[1]}') <(cat $trans | cut -d ' ' -f2-) > $dir/hixiaowen_text || exit 1
  for dataset in ticmini2_dataset_20180607 ticmini2_hixiaowen_adult_20180731 ticmini2_for_school_20180911; do
    cat data/download/$dataset/hixiaowen_text || exit 1
  done | sort -u -k1,1 > data/hixiaowen_text || exit 1
fi

if [ $stage -le 2 ]; then
  dir=data/download/ticmini2_dataset_20180607
  trans=$dir/freetext.txt
  paste -d ' ' <(cat $trans | awk '{print $1}' | awk '{split($1,a,"."); print a[1]}') <(cat $trans | cut -d ' ' -f2-) > $dir/freetext_text || exit 1
  dir=data/download/ticmini2_for_school_20180911
  trans=$dir/freetext/freetext.trans
  paste -d ' ' <(cat $trans | awk '{print $1}' | awk '{split($1,a,"/"); print a[4]}' | awk '{split($1,a,"."); print a[1]}') <(cat $trans | cut -d ' ' -f2-) > $dir/freetext_text || exit 1
  for dataset in ticmini2_dataset_20180607 ticmini2_for_school_20180911; do
    cat data/download/$dataset/freetext_text || exit 1
  done | sort -u -k1,1 > data/freetext_text || exit 1
fi

if [ $stage -le 3 ]; then
  dir=data/download/ticmini2_dataset_20180607
  trans=$dir/garbage.txt
  paste -d ' ' <(cat $trans | awk '{print $1}' | awk '{split($1,a,"."); print a[1]}') <(cat $trans | cut -d ' ' -f2-) > $dir/garbage_text || exit 1
  cat $dir/garbage_text > data/garbage_text
fi

if [ $stage -le 4 ]; then
  cat data/hixiaowen_text data/freetext_text data/garbage_text > data/text
  cat data/hixiaowen_text data/freetext_text | awk '{print $1}' | awk '{split($1,a,"-"); print $1,a[1]}' > data/hixiaowen_freetext_utt2spk || exit 1
  cat data/garbage_text | awk '{print $1}' | awk '{split($1,a,"_"); if(a[1]=="garbage") print $1,a[1] "_" a[2] "_" a[3]; else if(a[1]=="ticmini" || a[1]=="timini") print $1,a[1] "_" a[2] "_" a[3] "_" a[4] "_" a[5]; else print $1,$1}' | cat data/hixiaowen_freetext_utt2spk - > data/utt2spk || exit 1
  rm -f data/hixiaowen_freetext_utt2spk 2>/dev/null || true
fi

if [ $stage -le 5 ]; then
  for folder in train dev eval; do
    mkdir -p data/$folder
    utils/filter_scp.pl data/download/${folder}_list data/text > data/$folder/text || exit 1
    utils/filter_scp.pl data/download/${folder}_list data/utt2spk > data/$folder/utt2spk || exit 1
  done
fi

exit 0
