#!/usr/bin/env bash

# Copyright 2019 Xingyu Na
# Apache 2.0

. ./path.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <corpus-path> <data-path>"
  echo " $0 /export/a05/xna/data/stcmds data/stcmds"
  exit 1;
fi

corpus=$1/ST-CMDS-20170001_1-OS
data=$2

if [ ! -d $corpus ]; then
  echo "Error: $0 requires complete corpus"
  exit 1;
fi

echo "**** Creating ST-CMDS data folder ****"

mkdir -p $data/train

# find wav audio file for train

find $corpus -iname "*.wav" > $data/wav.list
n=`cat $data/wav.list | wc -l`
[ $n -ne 102600 ] && \
  echo Warning: expected 102600 data files, found $n

cat $data/wav.list | awk -F'20170001' '{print $NF}' | awk -F'.' '{print $1}' > $data/utt.list
cat $data/utt.list | awk '{print substr($1,1,6)}' > $data/spk.list

paste -d' ' $data/utt.list $data/wav.list > $data/train/wav.scp
paste -d' ' $data/utt.list $data/spk.list > $data/train/utt2spk

for file in wav.scp utt2spk; do
  sort $data/train/$file -o $data/train/$file
done

utils/utt2spk_to_spk2utt.pl $data/train/utt2spk > $data/train/spk2utt

rm -r $data/{wav,utt,spk}.list

utils/data/validate_data_dir.sh --no-feats --no-text $data/train || exit 1;
