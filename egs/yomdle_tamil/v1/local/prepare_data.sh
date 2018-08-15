#!/bin/bash

# Copyright      2018  Chun Chieh Chang
#                2018  Ashish Arora
#                2018  Hossein Hadian
# Apache 2.0

# This script prepares the training and test data (i.e text, images.scp,
# utt2spk and spk2utt) by calling process_data.py.

#  Eg. local/prepare_data.sh

#  Eg. text file: english_phone_books_0001_1 To sum up, then, it would appear that
#      utt2spk file: english_phone_books_0001_0 english_phone_books_0001
#      images.scp file: english_phone_books_0001_0 \
#      data/download/truth_line_image/english_phone_books_0001_0.png

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
  done
fi
