#!/bin/bash
# This script loads the IAM handwritten dataset

stage=0
nj=20
dir=data

. ./cmd.sh
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


mkdir -p $dir/{train,test}
if [ $stage -le 0 ]; then
  local/process_data.py $dir $dir/train --dataset train --model_type word || exit 1
  #local/process_data.py $dl_dir $dir/val_1 --dataset validationset1 --model_type word || exit 1
  #local/process_data.py $dl_dir $dir/val_2 --dataset validationset2 --model_type word || exit 1
  local/process_data.py $dir $dir/test --dataset test --model_type word || exit 1

  sort $dir/train/utt2spk -o $dir/train/utt2spk
  sort $dir/test/utt2spk -o $dir/test/utt2spk

  sort $dir/train/text -o $dir/train/text
  sort $dir/test/text -o $dir/test/text

  sort $dir/train/images.scp -o $dir/train/images.scp
  sort $dir/test/images.scp -o $dir/test/images.scp

  utils/utt2spk_to_spk2utt.pl $dir/train/utt2spk | sort -k 1 > $dir/train/spk2utt
  #utils/utt2spk_to_spk2utt.pl $dir/val_1/utt2spk > $dir/val_1/spk2utt
  #utils/utt2spk_to_spk2utt.pl $dir/val_2/utt2spk > $dir/val_2/spk2utt
  utils/utt2spk_to_spk2utt.pl $dir/test/utt2spk | sort -k 1 > $dir/test/spk2utt
fi
