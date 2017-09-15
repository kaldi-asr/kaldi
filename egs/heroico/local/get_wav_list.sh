#!/bin/bash

dir=$1

# heroico
find \
    "$dir/speech/heroico" \
    -type f \
    -name "*.wav" \
    | \
    sort \
	> \
	data/local/tmp/heroico/wav_list.txt

# USMA nonnative
if [ ! -d data/local/tmp/usma/nonnative ]; then
    mkdir -p data/local/tmp/usma/nonnative
fi

find \
    "$dir/speech/usma" \
    -type f \
    -name "*.wav" \
    | \
    grep \
	nonnative \
	| \
    sort \
	> \
	data/local/tmp/usma/nonnative/wav_list.txt

# USMA native

if [ ! -d data/local/tmp/usma/native ]; then
    mkdir -p data/local/tmp/usma/native
fi

find \
    "$dir/speech/usma" \
    -type f \
    -name "*.wav" \
    | \
    grep \
	-v \
	nonnative \
	| \
    sort \
	> \
	data/local/tmp/usma/native/wav_list.txt
