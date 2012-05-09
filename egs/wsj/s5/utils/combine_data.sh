#!/bin/bash
# Copyright 2012  Daniel Povey.  Apache 2.0.


if [ $# -le 2 ]; then
  echo "Usage: combine_data.sh <dest-data-dir> <src-data-dir1> <src-data-dir2> ..."
  exit 1
fi

dest=$1;
shift;

first_src=$1;

mkdir -p $dest;

export LC_ALL=C

for file in utt2spk feats.scp text cmvn.scp segments reco2file_and_channel wav.scp; do
  if [ -f $first_src/$file ]; then
    ( for f in $*; do cat $f/$file; done ) | sort -k1 > $dest/$file || exit 1;
    echo "$0: combined $file"
  else
    echo "$0 [info]: not combining $file as it does not exist"
  fi
done

utils/utt2spk_to_spk2utt.pl <$dest/utt2spk >$dest/spk2utt

utils/fix_data_dir.sh $dest || exit 1;

exit 0
