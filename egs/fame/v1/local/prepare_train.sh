#!/usr/bin/env bash
# Copyright 2015-2016  Sarah Flora Juan
# Copyright 2016  Johns Hopkins University (Author: Yenda Trmal)
# Copyright 2017  Radboud University (Author: Emre Yilmaz)

# Apache 2.0

corpus=$1
set -e -o pipefail
if [ -z "$corpus" ] ; then
    echo >&2 "The script $0 expects one parameter -- the location of the FAME! speech database"
    exit 1
fi
if [ ! -d "$corpus" ] ; then
    echo >&2 "The directory $corpus does not exist"
fi

mkdir -p data data/train

cp $corpus/data/spk2utt  data/train/spk2utt || exit 1;
cp $corpus/data/utt2spk  data/train/utt2spk || exit 1;

# the corpus wav.scp contains physical paths, so we just re-generate
# the file again from scratchn instead of figuring out how to edit it
for rec in $(awk '{print $1}' $corpus/data/utt2spk) ; do
    spk=${rec%_*}
    filename=$corpus/SD/${rec}.wav
    if [ ! -f "$filename" ] ; then
        echo >&2 "The file $filename could not be found ($rec)"
        exit 1
    fi
    # we might want to store physical paths as a general rule
    filename=$(utils/make_absolute.sh $filename)
    echo "$rec $filename"
done > data/train/wav.scp

# fix_data_dir.sh fixes common mistakes (unsorted entries in wav.scp,
# duplicate entries and so on). Also, it regenerates the spk2utt from
# utt2sp
utils/fix_data_dir.sh data/train
