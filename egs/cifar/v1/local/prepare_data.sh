#!/bin/bash

# Copyright 2017 Johns Hopkins University (author: Hossein Hadian)
# Apache 2.0

# This script loads the training and test data for CIFAR-10 or CIFAR-100.

[ -f ./path.sh ] && . ./path.sh; # source the path.

dl_dir=data/download
cifar10=$dl_dir/cifar-10-batches-bin
cifar10_url=https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
cifar100=$dl_dir/cifar-100-binary
cifar100_url=https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz

mkdir -p $dl_dir
if [ -d $cifar10 ]; then
  echo Not downloading CIFAR-10 as it is already there.
else
  if [ ! -f $dl_dir/cifar-10-binary.tar.gz ]; then
    echo Downloading CIFAR-10...
    wget -P $dl_dir $cifar10_url || exit 1;
  fi
  tar -xvzf $dl_dir/cifar-10-binary.tar.gz -C $dl_dir || exit 1;
  echo Done downloading and extracting CIFAR-10
fi

mkdir -p data/cifar10_{train,test}/data
seq 0 9 | paste -d' ' $cifar10/batches.meta.txt - | grep '\S' >data/cifar10_train/classes.txt
cp data/cifar10_{train,test}/classes.txt
echo 3 > data/cifar10_train/num_channels
echo 3 > data/cifar10_test/num_channels

local/process_data.py --dataset train $cifar10 data/cifar10_train/ | \
  copy-feats --compress=true --compression-method=7 \
   ark:- ark,scp:data/cifar10_train/data/images.ark,data/cifar10_train/images.scp || exit 1

local/process_data.py --dataset test $cifar10 data/cifar10_test/ | \
  copy-feats --compress=true --compression-method=7 \
    ark:- ark,scp:data/cifar10_test/data/images.ark,data/cifar10_test/images.scp || exit 1



### CIFAR 100

if [ -d $cifar100 ]; then
  echo Not downloading CIFAR-100 as it is already there.
else
  if [ ! -f $dl_dir/cifar-100-binary.tar.gz ]; then
    echo Downloading CIFAR-100...
    wget -P $dl_dir $cifar100_url || exit 1;
  fi
  tar -xvzf $dl_dir/cifar-100-binary.tar.gz -C $dl_dir || exit 1;
  echo Done downloading and extracting CIFAR-100
fi

mkdir -p data/cifar100_{train,test}/data
seq 0 99 | paste -d' ' $cifar100/fine_label_names.txt - | grep '\S' >data/cifar100_train/classes.txt

# seq 0 19 | paste -d' ' $cifar100/coarse_label_names.txt - | grep '\S' >data/cifar100_train/coarse_classes.txt

cp data/cifar100_{train,test}/classes.txt

#cp data/cifar100_{train,test}/coarse_classes.txt

echo 3 > data/cifar100_train/num_channels
echo 3 > data/cifar100_test/num_channels

local/process_data.py --cifar-version CIFAR-100 --dataset train $cifar100 data/cifar100_train/ | \
  copy-feats --compress=true --compression-method=7 \
    ark:- ark,scp:data/cifar100_train/data/images.ark,data/cifar100_train/images.scp || exit 1

local/process_data.py --cifar-version CIFAR-100 --dataset test $cifar100 data/cifar100_test/ | \
  copy-feats --compress=true --compression-method=7 \
    ark:- ark,scp:data/cifar100_test/data/images.ark,data/cifar100_test/images.scp || exit 1
