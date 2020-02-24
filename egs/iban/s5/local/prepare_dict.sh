#!/usr/bin/env bash
# Copyright 2015-2016  Sarah Flora Juan
# Copyright 2016  Johns Hopkins University (Author: Yenda Trmal)
# Apache 2.0

corpus=$1
if [ -z "$corpus" ] ; then
    echo >&2 "The script $0 expects one parameter -- the location of the Iban corpus"
    exit 1
fi
if [ ! -d "$corpus" ] ; then
    echo >&2 "The directory $corpus does not exist"
fi

mkdir -p data/lang data/local/dict


cp $corpus/lang/dict/lexicon.txt data/local/dict/lexicon.txt
cat data/local/dict/lexicon.txt | \
    perl -ane 'print join("\n", @F[1..$#F]) . "\n"; '  | \
    sort -u | grep -v 'SIL' > data/local/dict/nonsilence_phones.txt


touch data/local/dict/extra_questions.txt
touch data/local/dict/optional_silence.txt

echo "SIL"   > data/local/dict/optional_silence.txt
echo "SIL"   > data/local/dict/silence_phones.txt
echo "<UNK>" > data/local/dict/oov.txt

echo "Dictionary preparation succeeded"
