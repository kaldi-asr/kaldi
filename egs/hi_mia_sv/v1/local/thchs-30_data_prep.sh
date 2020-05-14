#!/usr/bin/env bash
# Copyright 2016  Tsinghua University (Author: Dong Wang, Xuewei Zhang).  Apache 2.0.
#           2016  LeSpeech (Author: Xingyu Na)

#This script pepares the data directory for thchs30 recipe. 
#It reads the corpus and get wav.scp and transcriptions.

corpus_dir=$1
data=$2

echo "**** Creating THCHS-30 data folder ****"
mkdir -p $data/{train,dev,test}

#create wav.scp, utt2spk.scp, spk2utt.scp, text
(
for x in train dev test; do
  echo "cleaning $data/$x"
  part=$data/$x
  rm -rf $part/{wav.scp,utt2spk,spk2utt}
  echo "preparing scps and text in $part"
  # updated new "for loop" figured out the compatibility issue with Mac     created by Xi Chen, in 03/06/2018
  for nn in `find  $corpus_dir/$x -name "*.wav" | sort -u | xargs -I {} basename {} .wav`; do
      spkid=`echo $nn | awk -F"_" '{print "" $1}'`
      spk_char=`echo $spkid | sed 's/\([A-Z]\).*/\1/'`
      spk_num=`echo $spkid | sed 's/[A-Z]\([0-9]\)/\1/'`
      spkid=$(printf '%s%.2d' "$spk_char" "$spk_num")
      utt_num=`echo $nn | awk -F"_" '{print $2}'`
      uttid=$(printf '%s%.2d_%.3d' "$spk_char" "$spk_num" "$utt_num" | sed 's/\_/\-/g')
      echo $uttid $corpus_dir/$x/$nn.wav >> $part/wav.scp
      echo $uttid $spkid >> $part/utt2spk
  done 
  sort $part/wav.scp -o $part/wav.scp
  sort $part/utt2spk -o $part/utt2spk
  utils/utt2spk_to_spk2utt.pl $part/utt2spk > $part/spk2utt
done
) || exit 1

utils/data/validate_data_dir.sh --no-feats --no-text $data/train || exit 1;
utils/data/validate_data_dir.sh --no-feats --no-text $data/dev || exit 1;
utils/data/validate_data_dir.sh --no-feats --no-text $data/test || exit 1;


