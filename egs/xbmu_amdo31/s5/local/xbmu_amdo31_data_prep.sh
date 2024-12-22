#!/usr/bin/env bash

# Copyright 2017 Xingyu Na
#           2021 Northwest Minzu University (senyan Li)
#Apache 2.0

. ./path.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <audio-path> <text-path>"
  echo " $0 /export/data/xbmu_amdo31/data/wav /export/data/xbmu_amdo31/data/transcript"
  exit 1;
fi

tibetan_audio_dir=$1
tibetan_text=$2/transcript_clean.txt

train_dir=data/local/train
dev_dir=data/local/dev
test_dir=data/local/test
tmp_dir=data/local/tmp

mkdir -p $train_dir
mkdir -p $dev_dir
mkdir -p $test_dir
mkdir -p $tmp_dir

# data directory check
if [ ! -d $tibetan_audio_dir ] || [ ! -f $tibetan_text ]; then
  echo "Error: $0 requires two directory arguments"
  exit 1;
fi
echo $tibetan_audio_dir
# find wav audio file for train, dev and test resp.
find $tibetan_audio_dir -iname "*.wav" > $tmp_dir/wav.flist
n=$(wc -l < "$tmp_dir/wav.flist")
[ $n -ne 22630 ] && \
  echo Warning: expected 141925 data data files, found $n

grep -i "wav/train" $tmp_dir/wav.flist > $train_dir/wav.flist || exit 1;
grep -i "wav/dev" $tmp_dir/wav.flist > $dev_dir/wav.flist || exit 1;
grep -i "wav/test" $tmp_dir/wav.flist > $test_dir/wav.flist || exit 1;

rm -r $tmp_dir
# Transcriptions preparation
# cat $tibetan_text |head -10
for dir in $train_dir $dev_dir $test_dir; do
  echo Preparing $dir transcriptions
  sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{print $NF}'  > $dir/utt.list
  sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{i=NF-1;printf("%s %s\n",$NF,$i)}'> $dir/utt2spk_all
  rm -f $dir/transcripts1.txt
  while read -r line
  do
      line1=$(echo "$line" | cut -d '-' -f 2)
      line2=$(grep -w $line1  $tibetan_text |cut -d " " -f 2-)
      text=$line" "$line2
      echo $text >>$dir/transcripts1.txt
  done < "$dir/utt.list"
  paste -d' ' $dir/utt.list $dir/wav.flist > $dir/wav.scp_all
  utils/filter_scp.pl -f 1 $dir/utt.list $dir/transcripts1.txt > $dir/transcripts.txt
  awk '{print $1}' $dir/transcripts.txt > $dir/utt.list
  utils/filter_scp.pl -f 1 $dir/utt.list $dir/utt2spk_all | sort -u > $dir/utt2spk
  utils/filter_scp.pl -f 1 $dir/utt.list $dir/wav.scp_all | sort -u > $dir/wav.scp
  sort -u $dir/transcripts.txt > $dir/text
  utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
done

mkdir -p data/train data/dev data/test

for f in spk2utt utt2spk wav.scp text; do
  cp $train_dir/$f data/train/$f || exit 1;
  cp $dev_dir/$f data/dev/$f || exit 1;
  cp $test_dir/$f data/test/$f || exit 1;
done

echo "$0: tibetan data preparation succeeded"
exit 0;
