#!/bin/bash

stage=0

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -eu


if [ $stage -le 1 ]; then
  dir=data/download/ticmini2_dataset_20180607
  trans=$dir/hixiaowen.txt
  paste -d ' ' <(cat $trans | awk '{print $1}' | awk '{split($1,a,"."); print a[1]}') <(cat $trans | cut -d ' ' -f2-) > $dir/hixiaowen_text || exit 1
  dir=data/download/ticmini2_hixiaowen_adult_20180731
  for folder in patch1 patch2; do
    trans=$dir/$folder/hixiaowen_trans
    paste -d ' ' <(cat $trans | awk '{print $1}' | awk '{split($1,a,"."); print a[1]}' | awk '{split($1,a,"/"); print a[3]}') <(cat $trans | cut -d ' ' -f2-) || exit 1
  done > $dir/hixiaowen_text || exit 1
  dir=data/download/ticmini2_for_school_20180911
  trans=$dir/hixiaowen/hixiaowen.trans
  paste -d ' ' <(cat $trans | awk '{print $1}' | awk '{split($1,a,"/"); print a[4]}' | awk '{split($1,a,"."); print a[1]}') <(cat $trans | cut -d ' ' -f2-) > $dir/hixiaowen_text || exit 1
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
    utils/filter_scp.pl local/${folder}_list data/text > data/$folder/text || exit 1
    utils/filter_scp.pl local/${folder}_list data/utt2spk > data/$folder/utt2spk || exit 1
  done
fi

if [ $stage -le 3 ] && false; then
  mkdir -p data/train
  python3 local/split_data_by_speakers.py data/hixiaowen_text data/freetext_text data/train/hixiaowen_freetext_text --train-proportion 0.8
  cat data/hixiaowen_text data/freetext_text | utils/filter_scp.pl --exclude data/train/hixiaowen_freetext_text | utils/shuffle_list.pl 2>/dev/null > data/dev_eval_hixiaowen_freetext_text || exit 1
  num_utts_dev=$(( $(wc -l data/dev_eval_hixiaowen_freetext_text | awk '{print $1}')/2 ))
  mkdir -p data/dev
  cat data/dev_eval_hixiaowen_freetext_text | head -$num_utts_dev > data/dev/hixiaowen_freetext_text
  mkdir -p data/eval
  cat data/dev_eval_hixiaowen_freetext_text | utils/filter_scp.pl --exclude data/dev/hixiaowen_freetext_text > data/eval/hixiaowen_freetext_text || exit 1
  rm -f data/dev_eval_hixiaowen_freetext_text 2>/dev/null || true
  for folder in train dev eval; do
    cat data/$folder/hixiaowen_freetext_text | awk '{print $1}' | awk '{split($1,a,"-"); print $1,a[1]}' > data/$folder/hixiaowen_freetext_utt2spk || exit 1
  done
fi

if [ $stage -le 4 ] && false; then
  dir=data/download/ticmini2_dataset_20180607
  trans=$dir/garbage.txt
  paste -d ' ' <(cat $trans | awk '{print $1}' | awk '{split($1,a,"."); print a[1]}') <(cat $trans | cut -d ' ' -f2-) > $dir/garbage_text || exit 1
  cat $dir/garbage_text > data/garbage_text
  num_utts_dev_eval=$(( $(wc -l data/garbage_text | awk '{print $1}')/5 ))
  cat data/garbage_text | utils/shuffle_list.pl 2>/dev/null | head -$num_utts_dev_eval > $dir/garbage_dev_eval_text || exit 1
  num_utts_dev=$(( ${num_utts_dev_eval}/2 ))
  cat $dir/garbage_dev_eval_text | head -$num_utts_dev > data/dev/garbage_text || exit 1
  utils/filter_scp.pl --exclude data/dev/garbage_text $dir/garbage_dev_eval_text > data/eval/garbage_text || exit 1
  cat data/garbage_text | utils/filter_scp.pl --exclude $dir/garbage_dev_eval_text > data/train/garbage_text || exit 1
  rm -f $dir/garbage_dev_eval_text 2>/dev/null || true
  for folder in train dev eval; do
    cat data/$folder/hixiaowen_freetext_text data/$folder/garbage_text > data/$folder/text
    cat data/$folder/garbage_text | awk '{print $1}' | awk '{split($1,a,"_"); if(a[1]=="garbage") print $1,a[1] "_" a[2] "_" a[3]; else if(a[1]=="ticmini" || a[1]=="timini") print $1,a[1] "_" a[2] "_" a[3] "_" a[4] "_" a[5]; else print $1,$1}' | cat data/$folder/hixiaowen_freetext_utt2spk - > data/$folder/utt2spk || exit 1
    rm -f data/$folder/hixiaowen_freetext_text data/$folder/garbage_text data/$folder/hixiaowen_freetext_utt2spk 2>/dev/null || true
  done
fi

