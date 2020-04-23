#!/bin/bash

# Copyright  2018-2020  Yiming Wang
#            2018-2020  Daniel Povey
# Apache 2.0

# This script loads the SNIPS dataset.
[ -f ./path.sh ] && . ./path.sh

dl_dir=data/download

mkdir -p $dl_dir

src_dir=/export/fs04/a07/ywang/snips-wake-word-corpus

dataset=hey_snips_kws_4.0.tar.gz
if [ -d $dl_dir/hey_snips_research_6k_en_train_eval_clean_ter ]; then
  echo "Not extracting $dl_dir/hey_snips_research_6k_en_train_eval_clean_ter) as it is already there."
else
  tar -xvzf $src_dir/$dataset -C $dl_dir || exit 1;
  echo "Done extracting $dataset."
fi

