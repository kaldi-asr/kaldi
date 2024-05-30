#!/usr/bin/env bash

## Only run this file from the example root directory
##      $ ./local/data_prep.sh

CORPUS_DIR="$1"

mkdir -p "data/local/dict"

source ./path.sh

#############################
# data/local/dict/lexicon.txt
#############################

export LC_ALL=C

echo -e '!SIL sil\n<UNK> spn' > data/local/dict/lexicon.txt
cat "$CORPUS_DIR/diccionarios/T22.full.dic" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -e 's/([0123456789]*)//g' \
        -e 's/\([^ ]\)n\~/\1n/g' \
        -e 's/a_7/a/g' -e 's/e_7/e/g' -e 's/i_7/i/g' -e 's/o_7/o/g' -e 's/u_7/u/g' \
        -e 's/a-7/a/g' -e 's/e-7/e/g' -e 's/i-7/i/g' -e 's/o-7/o/g' -e 's/u-7/u/g' \
        -e 's/a_/a/g' -e 's/e_/e/g' -e 's/i_/i/g' -e 's/o_/o/g' -e 's/u_/u/g' \
    | sed -e 's/_7n.*$//' \
        -e 's/atl_7tica/atletica/' \
        -e 's/biol_7gicas/biologicas/' \
        -e 's/elec_7ctrico/electrico/' \
        -e 's/gr_7afico/grafico/' \
        -e 's/s_7lo/solo/' \
    | sed -e 's/n~/ni/g' -e 's/r(/rh/g' \
    | sed -e 's/\t/ /g' -e '/^$/d' \
    | sort | uniq \
    >> data/local/dict/lexicon.txt


#######################################
# data/local/dict/silence_phones.txt
# data/local/dict/optional_silence.txt
# data/local/dict/nonsilence_phones.txt
# data/local/dict/extra_questions.txt
#######################################

echo -e 'sil\nspn' > data/local/dict/silence_phones.txt
echo -e 'sil' > data/local/dict/optional_silence.txt
cat data/local/dict/lexicon.txt \
    | grep -v '<UNK>' \
    | grep -v '!SIL' \
    | cut -d' ' -f1 --complement \
    | sed 's/ /\n/g' \
    | sort -u \
    > data/local/dict/nonsilence_phones.txt
