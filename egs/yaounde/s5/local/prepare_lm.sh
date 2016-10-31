#!/bin/bash
. ./cmd.sh
. ./path.sh
mkdir -p data/local/lm
corpus=$1
lmthreegram=data/local/lm/lm_threegram.arpa
ngram-count -order 3 -kndiscount -interpolate -unk -map-unk "<UNK>" -limit-vocab -text $corpus -lm $lmthreegram || exit 1
if [ -e "data/local/lm/lm_threegram.arpa.gz" ]; then
    rm data/local/lm/lm_threegram.arpa.gz
fi
gzip  data/local/lm/lm_threegram.arpa
