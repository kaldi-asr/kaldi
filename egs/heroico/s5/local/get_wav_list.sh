#!/bin/bash

# Copyright 2017 John Morgan
# Apache 2.0.

dir=$1

tmpdir=data/local/tmp/heroico

mkdir -p $tmpdir

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
