#!/usr/bin/env bash

## Install SRILM in the `tools` directory (install_srilm.sh)

## Only run this file from the example root directory
##      $ ./local/data_prep.sh

mkdir -p "data/local/tmp" "data/lang/tmp"

source ./path.sh

if [ -d "../../../tools/srilm/bin/i686-m64" ]; then
    ngram_count_exe="../../../tools/srilm/bin/i686-m64/ngram-count"
elif [ -d "../../../tools/srilm/bin/i686" ]; then
    ngram_count_exe="../../../tools/srilm/bin/i686/ngram-count"
else
    echo
    echo "[!] Install SRILM in the 'tools' directory (install_srilm.sh)"
    echo
    exit 1
fi


########################
# data/local/tmp/lm_text
########################

# Text sentences input for language model generation
# taken from data/[train|test]/text but with utterance IDs removed

cat data/train/text data/test/text | cut -d' ' -f1 --complement > data/local/tmp/lm_text


#################################
# data/local/tmp/3gram_arpa_lm.gz
##################################

$ngram_count_exe -lm data/local/tmp/3gram_lm.arpa.kn.gz \
    -order 3 \
    -write-vocab data/local/tmp/vocab-full.txt \
    -sort \
    -wbdiscount \
    -unk \
    -map-unk "<UNK>" \
    -text data/local/tmp/lm_text
    # -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 2 \
    # -kndiscount3 -gt3min 3 -order 3 \


#################
# data/lang/G.fst
#################

utils/format_lm.sh data/lang \
    data/local/tmp/3gram_lm.arpa.kn.gz \
    data/local/dict/lexicon.txt \
    data/lang
