#!/bin/bash

# Copyright 2017 Johns Hopkins University (author: Hossein Hadian)
# Apache 2.0

# This script loads the training and test data for SVHN
# (Street View House Numbers) dataset.

[ -f ./path.sh ] && . ./path.sh; # source the path.

dl_dir=data/download
base_url=http://ufldl.stanford.edu/housenumbers

test_url=http://ufldl.stanford.edu/housenumbers/test_32x32.mat
train_url=http://ufldl.stanford.edu/housenumbers/train_32x32.mat
extra_url=http://ufldl.stanford.edu/housenumbers/extra_32x32.mat

mkdir -p $dl_dir
mkdir -p data/{train,test,extra}/data

for datafile in test_32x32.mat train_32x32.mat extra_32x32.mat; do
  url=$base_url/$datafile
  if [ -f $dl_dir/$datafile ]; then
    echo Not downloading $datafile as it is already there.
  else
    echo Downloading $datafile...
    wget -P $dl_dir $url || exit 1;
  fi
  out_data_dir=$(echo $datafile | cut -d'_' -f 1)
  local/process_data.py $dl_dir/$datafile data/$out_data_dir/ | \
    copy-feats --compress=true --compression-method=7 \
     ark:- ark,scp:data/$out_data_dir/data/images.ark,data/$out_data_dir/images.scp || exit 1
done

seq 0 9 | awk '{print $1 " " $1}' > data/train/classes.txt

cp data/{train,test}/classes.txt
cp data/{train,extra}/classes.txt

echo 3 > data/train/num_channels
echo 3 > data/test/num_channels
echo 3 > data/extra/num_channels

# prepare train_all data dir (i.e. train+extra)
mkdir data/train_all
cp data/{train,train_all}/classes.txt
cp data/{train,train_all}/num_channels
cat data/train/images.scp | awk '{print "t_" $0}' > data/train_all/images.scp
cat data/train/labels.txt | awk '{print "t_" $0}' > data/train_all/labels.txt
cat data/extra/images.scp | awk '{print "e_" $0}' >> data/train_all/images.scp
cat data/extra/labels.txt | awk '{print "e_" $0}' >> data/train_all/labels.txt

