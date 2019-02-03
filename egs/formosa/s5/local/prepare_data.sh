#!/bin/bash
# Copyright 2015-2016  Sarah Flora Juan
# Copyright 2016  Johns Hopkins University (Author: Yenda Trmal)
# Copyright 2018  Yuan-Fu Liao, National Taipei University of Technology
#                 AsusTek Computer Inc. (Author: Alex Hung)

# Apache 2.0

set -e -o pipefail

corpus=NER-Trs-Vol1/Train

. ./path.sh
. parse_options.sh

if [ ! -d "$corpus" ] ; then
    echo >&2 "The directory $corpus does not exist"
fi

if [ -z "$(command -v dos2unix 2>/dev/null)" ]; then
    echo "dos2unix not found on PATH. Please install it manually."
    exit 1;
fi

# have to remvoe previous files to avoid filtering speakers according to cmvn.scp and feats.scp
rm -rf   data/all data/train data/test data/local/train
mkdir -p data/all data/train data/test data/local/train


# make utt2spk, wav.scp and text
find $corpus -name *.wav -exec sh -c 'x={}; y=${x%.wav}; printf "%s %s\n"     $y $y' \; | dos2unix > data/all/utt2spk
find $corpus -name *.wav -exec sh -c 'x={}; y=${x%.wav}; printf "%s %s\n"     $y $x' \; | dos2unix > data/all/wav.scp
find $corpus -name *.txt -exec sh -c 'x={}; y=${x%.txt}; printf "%s " $y; cat $x'    \; | dos2unix | sed 's:/Text/:/Wav/:g' > data/all/text

# fix_data_dir.sh fixes common mistakes (unsorted entries in wav.scp,
# duplicate entries and so on). Also, it regenerates the spk2utt from
# utt2spk
utils/fix_data_dir.sh data/all

echo "Preparing train and test data"
# test set: JZ, GJ, KX, YX
grep -E "Wav/(JZ|GJ|KX|YX)" data/all/utt2spk | awk '{print $1}' > data/all/cv.spk
utils/subset_data_dir_tr_cv.sh --cv-spk-list data/all/cv.spk data/all data/train data/test

# for LM training
echo "cp data/train/text data/local/train/text for language model training"
cat data/train/text | awk '{$1=""}1;' | awk '{$1=$1}1;' > data/local/train/text

echo "Data preparation completed."
exit 0;
