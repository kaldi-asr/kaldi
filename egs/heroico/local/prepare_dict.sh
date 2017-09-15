#!/bin/bash -u

set -o errexit

[ -f path.sh ] && . path.sh  # Sets the PATH to contain necessary executables

wget http://www.openslr.org/resources/34/santiago.tar.gz

mkdir -p local/src/dict

mv santiago.tar.gz local/src/dict/

if [ -e local/src/dict/santiago.tar ]; then
    rm local/src/dict/santiago.tar
fi

gunzip local/src/dict/santiago.tar.gz

cd local/src/dict

tar -xvf santiago.tar

cd ../../..

if [ ! -d data/local/dict ]; then
    mkdir -p data/local/dict
fi

export LC_ALL=C

sort \
    local/src/dict/santiago.txt \
    | \
    uniq \
    | \
    sed "1d" \
	> \
	data/local/dict/lexicon.txt

echo "<UNK>	SPN" \
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

cut \
    -f2- \
    -d "	" \
    data/local/dict/lexicon.txt \
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

(
    tr '\n' ' ' < data/local/dict/silence_phones.txt;
    echo;
    tr '\n' ' ' < data/local/dict/nonsilence_phones.txt;
    echo;
) >data/local/dict/extra_questions.txt

echo "Finished dictionary preparation."
