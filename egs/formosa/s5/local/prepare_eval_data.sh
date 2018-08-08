#!/bin/bash
# Copyright 2015-2016  Sarah Flora Juan
# Copyright 2016  Johns Hopkins University (Author: Yenda Trmal)
# Apache 2.0

set -e -o pipefail

corpus=NER-Trs-Vol1-Eval/

if [ -z "$corpus" ] ; then
    echo >&2 "The script $0 expects one parameter -- the location of the speech corpus"
    exit 1
fi
if [ ! -d "$corpus" ] ; then
    echo >&2 "The directory $corpus does not exist"
fi

echo "Preparing eval data"

# have to remvoe previous files to avoid filtering speakers according to cmvn.scp and feats.scp
rm -rf   data/eval
mkdir -p data/eval

#
# make utt2spk, wav.scp and text
#

touch data/eval/utt2spk
touch data/eval/wav.scp
touch data/eval/text
find $corpus -name *.wav -exec sh -c 'x={}; y=${x%.wav}; printf "%s %s\n"     $y $y' \; > data/eval/utt2spk
find $corpus -name *.wav -exec sh -c 'x={}; y=${x%.wav}; printf "%s %s\n"     $y $x' \; > data/eval/wav.scp
find $corpus -name *.txt -exec sh -c 'x={}; y=${x%.txt}; printf "%s " $y; cat $x'    \; | sed 's/\/Text\//\/Wav\//' | sed 's/\/Submission\/NER-Trs-Vol1-Eval-Submission//' > data/eval/text

#
# fix data format
#

utils/fix_data_dir.sh data/eval

echo "Eval Data preparation completed."

