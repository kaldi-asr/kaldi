#!/bin/bash -u

# Copyright 2018 John Morgan
# Apache 2.0.

set -o errexit

[ -f ./path.sh ] && . ./path.sh

l=$1
dir=$2

if [ ! -d $dir ]; then
  mkdir -p $dir
fi


export LC_ALL=C

cut -f2- -d " " $l | tr -s '[:space:]' '[\n*]' | grep -v SPN | \
    sort -u > $dir/nonsilence_phones.txt

expand -t 1 $l | sort -u | \
    sed s/\([23456789]\)// | \
    sed s/\(1[0123456789]\)// | \
    sort -u > $dir/lexicon.txt

# silence phones, one per line.
{
    echo SIL;
    echo SPN;
} \
    > \
    $dir/silence_phones.txt

echo SIL > $dir/optional_silence.txt

# get the phone list from the lexicon file
(
    tr '\n' ' ' < $dir/silence_phones.txt;
    echo;
    tr '\n' ' ' < $dir/nonsilence_phones.txt;
    echo;
) >$dir/extra_questions.txt

echo "$0: Finished dictionary preparation."
