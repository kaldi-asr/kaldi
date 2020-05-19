#!/usr/bin/env bash

# Copyright 2017  Vimal Manohar
# Apache 2.0.

# Copy the data directory, but modify it to use the recording-id as the 
# speaker. This is useful to get matching speaker information in the 
# whole recording data directory.
# Note that this also appends the recording-id as a prefix to the 
# utterance-id.

if [ $# -ne 2 ]; then
  echo "Usage: $0 <in-data> <out-data>"
  echo " e.g.: $0 data/train data/train_recospk"
  exit 1
fi

in_data=$1
out_data=$2

mkdir -p $out_data

for f in wav.scp segments utt2spk; do 
  if [ ! -f $in_data/$f ]; then
    echo "$0: Could not find file $in_data/$f" 
    exit 1
  fi
done

cp $in_data/wav.scp $out_data/ || exit 1
cp $in_data/reco2file_and_channel $out_data/ 2> /dev/null || true
awk '{print $1" "$2"-"$1}' $in_data/segments > \
  $out_data/old2new.uttmap || exit 1
utils/apply_map.pl -f 1 $out_data/old2new.uttmap < $in_data/segments > \
  $out_data/segments || exit 1
awk '{print $1" "$2}' $out_data/segments > $out_data/utt2spk || exit 1
utils/utt2spk_to_spk2utt.pl $out_data/utt2spk > $out_data/spk2utt || exit 1

if [ -f $in_data/text ]; then
  utils/apply_map.pl -f 1 $out_data/old2new.uttmap < $in_data/text > \
    $out_data/text || exit 1
fi

if [ -f $in_data/feats.scp ]; then
  utils/apply_map.pl -f 1 $out_data/old2new.uttmap < $in_data/feats.scp > \
    $out_data/feats.scp || exit 1
fi

utils/fix_data_dir.sh $out_data || exit 1
utils/validate_data_dir.sh --no-text --no-feats $out_data || exit 1
