#!/bin/bash -u

# Copyright 2017 John Morgan
# Apache 2.0.

set -o errexit

[ -f ./path.sh ] && . ./path.sh

if [ ! -d data/local/dict ]; then
    mkdir -p data/local/dict
fi

export LC_ALL=C

cut \
    -f2- \
    -d "	" \
    data/local/tmp/dict/santiago.txt \
    | \
    tr -s '[:space:]' '[\n*]' \
    | \
    grep \
	-v \
	SPN \
    | \
        sort \
    | \
    uniq \
	> \
	data/local/dict/nonsilence_phones.txt

expand \
    -t 1 \
    data/local/tmp/dict/santiago.txt \
    | \
    sort \
    | \
    uniq \
    | \
    sed "1d" \
	> \
	data/local/dict/lexicon.txt

echo "<UNK> SPN" \
     >> \
	data/local/dict/lexicon.txt

# silence phones, one per line.
{
    echo SIL;
    echo SPN;
} \
    > \
    data/local/dict/silence_phones.txt

echo \
    SIL \
    > \
    data/local/dict/optional_silence.txt

(
    tr '\n' ' ' < data/local/dict/silence_phones.txt;
    echo;
    tr '\n' ' ' < data/local/dict/nonsilence_phones.txt;
    echo;
) >data/local/dict/extra_questions.txt

echo "Finished dictionary preparation."
