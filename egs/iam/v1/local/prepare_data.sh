#!/bin/bash

[ -f ./path.sh ] && . ./path.sh;

# various file paths
dl_dir=data/download

mkdir -p data/{train,val_1,val_2,test}/data

local/process_feature_vect.py $dl_dir data/train \
  --dataset trainset --scale-size 40 | \
  copy-feats --compress=true --compression-method=7 \
  ark:- ark,scp:data/train/data/images.ark,data/train/images.scp || exit 1
