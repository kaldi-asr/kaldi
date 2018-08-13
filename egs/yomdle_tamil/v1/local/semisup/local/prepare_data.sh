#!/bin/bash

stage=0
language=tamil

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1;

mkdir -p data/{train,test,train_unsup}
if [ $stage -le 1 ]; then
  local/process_data.py data/download/${language}/ \
    data/local/splits/yomdle-${language}-train_unsup.list \
    data/train_unsup
  image/fix_data_dir.sh data/train_unsup
  rm -rf data/train_unsup/.backup
fi
