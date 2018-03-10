#!/bin/bash
. ./cmd.sh
. ./path.sh

# To be run from one directory above this script.
# Creat text, utt2spk, spk2utt, images.scp, and feats.scp for test and train.

# oldLC should be some utf8.*
oldLC=en_US.utf8

database='/export/b01/babak/IFN-ENIT/ifnenit_v2.0p1e/data'

# source ./path.sh

mkdir -p data

export LC_ALL=$oldLC


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

  local/process_data.py $database data/$set --dataset $set --model_type word || exit 1
  sort data/$set/images.scp -o data/$set/images.scp
  sort data/$set/utt2spk -o data/$set/utt2spk

  utils/utt2spk_to_spk2utt.pl data/$set/utt2spk > data/$set/spk2utt

  mkdir -p data/{train,test}/data

  local/make_feature_vect.py data/$set --scale-size 40 | \
    copy-feats --compress=true --compression-method=7 \
    ark:- ark,scp:data/$set/data/images.ark,data/$set/feats.scp || exit 1

  steps/compute_cmvn_stats.sh data/$set || exit 1;

done

export LC_ALL=C


