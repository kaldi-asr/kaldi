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

echo "copying $srcdir to ${outdir}_aug1"
utils/copy_data_dir.sh --spk-prefix aug1- --utt-prefix aug1- $srcdir ${outdir}_aug1

echo " calling local/extract_features.sh for extracting features"
local/extract_features.sh --nj $nj --cmd "$cmd" --feat-dim $feat_dim --fliplr false --augment true ${outdir}_aug1

echo " combine original data and data from different augmentations"
utils/combine_data.sh --extra-files images.scp $outdir  $srcdir ${outdir}_aug1

# remove the temporary augX dir's:
rm -r ${outdir}_aug1
