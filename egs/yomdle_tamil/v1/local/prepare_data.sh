#!/bin/bash

stage=0
language=tamil

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1;

mkdir -p data/{train,test,train_unsup}
if [ $stage -le 1 ]; then
    for datasplit in train test train_unsup; do
        local/process_data.py data/download/${language}/ \
            data/local/splits/yomdle-${language}-${datasplit}.list \
            data/${datasplit}
        image/fix_data_dir.sh data/${datasplit}
        rm -rf data/${datasplit}/.backup
    done
fi
