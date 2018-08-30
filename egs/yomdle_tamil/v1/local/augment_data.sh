#!/bin/bash
# Copyright   2018 Hossein Hadian
#             2018 Ashish Arora

# Apache 2.0
# This script performs data augmentation.

nj=4
cmd=run.pl
feat_dim=40
echo "$0 $@"

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1;

srcdir=$1
outdir=$2
datadir=$3

mkdir -p $datadir/augmentations
echo "copying $srcdir to $datadir/augmentations/aug1"
utils/copy_data_dir.sh --spk-prefix aug1- --utt-prefix aug1- $srcdir $datadir/augmentations/aug1

echo " copying allowed length for training with augmented data..."
cat $srcdir/allowed_lengths.txt > $datadir/augmentations/aug1/allowed_lengths.txt

echo " Extracting features, creating feats.scp file for augmentated data"
local/extract_features.sh --nj $nj --cmd "$cmd" --feat-dim $feat_dim --fliplr false --augment true $datadir/augmentations/aug1

echo " combine original data and data from different augmentations"
utils/combine_data.sh --extra-files images.scp $outdir $srcdir $datadir/augmentations/aug1
cat $srcdir/allowed_lengths.txt > $outdir/allowed_lengths.txt
