#!/bin/bash

stage=0
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1;

if [ $stage -le 1 ]; then
  for datasplit in train test; do
    local/process_data.py data/download/ \
      data/local/splits/${datasplit}.txt \
      data/${datasplit}
    image/fix_data_dir.sh data/${datasplit}
    rm -rf data/${datasplit}/.backup
  done
fi
