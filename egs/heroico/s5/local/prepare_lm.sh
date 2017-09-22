#!/bin/bash

# Copyright 2017 John Morgan
# Apache 2.0.

. cmd.sh
set -e
. ./path.sh
stage=0

. ./utils/parse_options.sh

if [ $stage -le 0 ]; then
    mkdir -p language_models \
	data/local/lm

    # use only training prompts
    (cut -f 2 data/train/text > data/local/tmp/lm_training_text.txt)
    corpus=data/local/tmp/lm_training_text.txt

    ngram-count \
	-order 3 \
	-interpolate \
	-unk \
	-map-unk "<UNK>" \
	-limit-vocab \
	-text $corpus \
	-lm language_models/lm_threegram.arpa || exit 1;

    if [ -e "language_models/lm_threegram.arpa.gz" ]; then
	rm language_models/lm_threegram.arpa.gz
    fi

    gzip \
	language_models/lm_threegram.arpa
fi
