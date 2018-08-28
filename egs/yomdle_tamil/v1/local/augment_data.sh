#!/bin/bash
# Copyright   2018 Hossein Hadian
#             2018 Ashish Arora

nj=4
cmd=run.pl
feat_dim=40
echo "$0 $@"

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1;

src_dir=$1
out_dir=$2

echo "copying $src_dir to ${outdir}_aug1"
utils/copy_data_dir.sh --spk-prefix aug1- --utt-prefix aug1- $srcdir ${outdir}_aug1

echo " calling local/extract_features.sh for extracting features"
local/extract_features.sh --nj $nj --cmd "$cmd" --feat-dim $feat_dim --fliplr $fliplr --augment true ${outdir}_aug1

#utils/combine_data_dir.sh --extra-files images.scp $outdir  $srcdir ${outdir}_aug1 ${outdir}_aug2 ...
