#!/bin/bash

# To be run from one directory above this script.
# Creat text, utt2spk, spk2utt, images.scp, and feats.scp for test and train.

database_dir= # directory of the dataset
train_sets= # sets for training
test_sets=  # sets for testing

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

mkdir -p data
for set in 'train' 'test'
do
  ## Clean up
  if [[ -f tmp.unsorted ]]
  then
    rm tmp.unsorted
  fi
  if [ -d "data/$set" ]; then
    rm -r data/$set
  fi

  ## Gather transcriptions
  mkdir data/$set
  cat data/text.$set > tmp.unsorted
  # done
  cat tmp.unsorted | sort -k1 > tmp.sorted
  cat tmp.sorted | cut -d' ' -f1 > data/$set/uttids
  cat tmp.sorted | cut -d' ' -f2- | python3 local/remove_diacritics.py | python3 local/replace_arabic_punctuation.py | tr '+' '\\' | tr '=' '\\' | sed 's/\xA0/X/g' | sed 's/\x00\xA0/X/g' | sed 's/\xC2\xA0/X/g' | sed 's/\s\+/ /g' | sed 's/ \+$//' | sed 's/^ \+$//' | paste -d' ' data/$set/uttids - > data/$set/text
  rm tmp.unsorted tmp.sorted

  local/process_data.py $database_dir data/$set --dataset $set --train_sets "$train_sets" --test_sets "$test_sets" || exit 1
  sort data/$set/images.scp -o data/$set/images.scp
  sort data/$set/utt2spk -o data/$set/utt2spk

  utils/utt2spk_to_spk2utt.pl data/$set/utt2spk > data/$set/spk2utt

  mkdir -p data/{train,test}/data

  local/make_features.py data/$set --feat-dim 40 | \
    copy-feats --compress=true --compression-method=7 \
    ark:- ark,scp:data/$set/data/images.ark,data/$set/feats.scp || exit 1

  steps/compute_cmvn_stats.sh data/$set || exit 1;

done
