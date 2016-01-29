#!/bin/bash

dir=$1
corpus_dir=$2

#clean thchs30 corpus
#get wav.scp and transcriptions from thchs30 corpus

cd $dir

echo "creating data/{train,dev,test}"
mkdir -p data/{train,dev,test}

#create wav.scp, utt2spk.scp, spk2utt.scp, text
(
for x in train dev test; do
  echo "cleaning data/$x"
  cd $dir/data/$x
  rm -rf wav.scp utt2spk spk2utt word.txt phone.txt text
  echo "preparing scps and text in data/$x"
  for nn in `find  $corpus_dir/$x/*.wav | sort -u | xargs -i basename {} .wav`; do
      echo $nn $corpus_dir/$x/$nn.wav >> wav.scp
      echo $nn $nn >> utt2spk
      echo $nn $nn >> spk2utt
      echo $nn `sed -n 1p $corpus_dir/data/$nn.wav.trn` >> word.txt
      echo $nn `sed -n 3p $corpus_dir/data/$nn.wav.trn` >> phone.txt
  done 
  cp word.txt text
done
) || exit 1

echo "creating test_phone for phone decoding"
(
  rm -rf data/test_phone && cp -R data/test data/test_phone  || exit 1
  cd data/test_phone && rm text &&  cp phone.txt text || exit 1
)

