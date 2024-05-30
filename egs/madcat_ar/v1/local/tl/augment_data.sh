#!/usr/bin/env bash
# Copyright   2018 Hossein Hadian
#             2018 Ashish Arora

# Apache 2.0
# This script performs data augmentation.

nj=4
cmd=run.pl
feat_dim=40
verticle_shift=0
echo "$0 $@"

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1;

srcdir=$1
outdir=$2
datadir=$3
aug_set=aug1
mkdir -p $datadir/augmentations
echo "copying $srcdir to $datadir/augmentations/$aug_set, allowed length, creating feats.scp"

for set in $aug_set; do
  image/copy_data_dir.sh --spk-prefix $set- --utt-prefix $set- \
    $srcdir $datadir/augmentations/$set
  cat $srcdir/allowed_lengths.txt > $datadir/augmentations/$set/allowed_lengths.txt
  local/extract_features.sh --nj $nj --cmd "$cmd" --feat-dim $feat_dim \
    --vertical-shift $verticle_shift \
    --augment 'random_shift' $datadir/augmentations/$set
done

echo " combine original data and data from different augmentations"
utils/combine_data.sh --extra-files images.scp $outdir $srcdir $datadir/augmentations/$aug_set
cat $srcdir/allowed_lengths.txt > $outdir/allowed_lengths.txt
