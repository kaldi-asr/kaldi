#!/usr/bin/env bash
# Copyright 2019 QCRI (Author: Ahmed Ali)
# Apache 2.0

set -e -o pipefail


###
# The script assumes you have downloaded to the MGB-5 corpus: https://arabicspeech.org/mgb5
# DB/{dev.tar.gz,train.tar.gz}
###
echo "Preparing train and dev data"

if [[ ! -e "DB/train.tar.gz" || ! -e "DB/dev.tar.gz" ]]; then
  echo "You need to download the MGB-5 first and copy dev.tar.gz and train.tar.gz to DB folder"
  echo "check: https://arabicspeech.org/mgb5"
  exit 1
fi

# We will extract data again even if you did this before.
(cd DB; rm -fr train dev;for x in *; do tar -xvf $x; done)

mkdir -p data/local data/train data/dev

for x in train dev; do
    sed -e 's:UNK: :g' -e 's:  : :g' DB/$x/$x.txt.bw > data/$x/text #removing words that annotators couldn't understand
    cp DB/$x/$x.segments.bw data/$x/segments
    awk '{print $1 " " $1}' DB/$x/$x.segments.bw > data/$x/spk2utt
    cp data/$x/spk2utt data/$x/utt2spk 
    find $PWD/DB/$x/ -name \*.wav | while read wav; do
        id=$(basename $wav | sed 's:.wav::')
        echo $id $wav
    done | sort -u > data/$x/wav.scp
    utils/fix_data_dir.sh data/$x
done


echo "Data preparation completed."

