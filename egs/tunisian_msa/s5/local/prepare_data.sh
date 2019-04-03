#!/bin/bash  

# Copyright 2018 John Morgan
# Apache 2.0.

# configuration variables
tmpdir=data/local/tmp
download_dir=$(pwd)
tmp_tunis=$tmpdir/tunis
tmp_libyan=$tmpdir/libyan
data_dir=$download_dir/Tunisian_MSA/data
# location of test data 
libyan_src=$data_dir/speech/test/Libyan_MSA
# end of configuration variable settings

# process the Tunisian MSA devtest data

# get list of  wav files
for s in devtest/CTELLONE/Recordings_Arabic/6 devtest/CTELLTHREE/Recordings_Arabic/10; do
  echo "$0: looking for wav files for $s."
  mkdir -p $tmp_tunis/$s
  find $data_dir/speech/$s -type f \
  -name "*.wav" | grep Recordings_Arabic > $tmp_tunis/$s/wav.txt

  local/devtest_recordings_make_lists.pl \
  $data_dir/transcripts/devtest/recordings.tsv $s tunis

  mkdir -p data/devtest

  for x in wav.scp utt2spk text; do
    cat     $tmp_tunis/$s/$x | tr "	" " " >> data/devtest/$x
  done
done

utils/utt2spk_to_spk2utt.pl data/devtest/utt2spk | sort > data/devtest/spk2utt

utils/fix_data_dir.sh data/devtest

# training data consists of 2 parts: answers and recordings (recited)
answers_transcripts=$data_dir/transcripts/train/answers.tsv
recordings_transcripts=$data_dir/transcripts/train/recordings.tsv

# location of test data
cls_rec_tr=$libyan_src/cls/data/transcripts/recordings/cls_recordings.tsv
lfi_rec_tr=$libyan_src/lfi/data/transcripts/recordings/lfi_recordings.tsv
srj_rec_tr=$libyan_src/srj/data/transcripts/recordings/srj_recordings.tsv
mbt_rec_tr=$data_dir/transcripts/test/mbt/recordings/mbt_recordings.tsv

# make acoustic model training  lists
mkdir -p $tmp_tunis

# get  wav file names

# for recited speech
# the data collection laptops had names like CTELLONE CTELLTWO ...
for machine in CTELLONE CTELLTWO CTELLTHREE CTELLFOUR CTELLFIVE; do
  find $data_dir/speech/train/$machine -type f -name "*.wav" | grep Recordings \
  >> $tmp_tunis/recordings_wav.txt
done

# get file names for Answers 
for machine in CTELLONE CTELLTWO CTELLTHREE CTELLFOUR CTELLFIVE; do
  find $data_dir/speech/train/$machine -type f \
    -name "*.wav" \
    | grep Answers >> $tmp_tunis/answers_wav.txt
done

# make separate transcription lists for answers and recordings
export LC_ALL=en_US.UTF-8
local/answers_make_lists.pl $answers_transcripts

utils/fix_data_dir.sh $tmp_tunis/answers

local/recordings_make_lists.pl $recordings_transcripts

utils/fix_data_dir.sh $tmp_tunis/recordings

# consolidate lists
# acoustic models will be trained on both recited and prompted speech
mkdir -p $tmp_tunis/lists

for x in wav.scp utt2spk text; do
  cat $tmp_tunis/answers/$x $tmp_tunis/recordings/$x > $tmp_tunis/lists/$x
done

utils/fix_data_dir.sh $tmp_tunis/lists

# get training lists
mkdir -p data/train
for x in wav.scp utt2spk text; do
  sort $tmp_tunis/lists/$x | tr "	" " " > data/train/$x
done

utils/utt2spk_to_spk2utt.pl data/train/utt2spk | sort > data/train/spk2utt

utils/fix_data_dir.sh data/train

# process the Libyan MSA data
mkdir -p $tmp_libyan

for s in cls lfi srj; do
  mkdir -p $tmp_libyan/$s

  # get list of  wav files
  find $libyan_src/$s -type f \
    -name "*.wav" \
    | grep recordings > $tmp_libyan/$s/recordings_wav.txt

  echo "$0: making recordings list for $s"
  local/test_recordings_make_lists.pl \
    $libyan_src/$s/data/transcripts/recordings/${s}_recordings.tsv $s libyan
done

# process the Tunisian MSA test data

mkdir -p $tmp_tunis/mbt

# get list of  wav files
find $data_dir/speech/test/mbt -type f \
  -name "*.wav" \
  | grep recordings > $tmp_tunis/mbt/recordings_wav.txt

echo "$0: making recordings list for mbt"
local/test_recordings_make_lists.pl \
  $data_dir/transcripts/test/mbt/recordings/mbt_recordings.tsv mbt tunis

mkdir -p data/test
# get the Libyan files
for s in cls lfi srj; do
  for x in wav.scp utt2spk text; do
    cat     $tmp_libyan/$s/recordings/$x | tr "	" " " >> data/test/$x
  done
done

for x in wav.scp utt2spk text; do
  cat     $tmp_tunis/mbt/recordings/$x | tr "	" " " >> data/test/$x
done

utils/utt2spk_to_spk2utt.pl data/test/utt2spk | sort > data/test/spk2utt

utils/fix_data_dir.sh data/test
