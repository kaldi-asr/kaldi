#!/usr/bin/env bash
# Copyright 2015-2016  Sarah Flora Juan
# Copyright 2016  Johns Hopkins University (Author: Yenda Trmal)
# Apache 2.0

corpus=$1
set -e -o pipefail
if [ -z "$corpus" ] ; then
    echo >&2 "The script $0 expects one parameter -- the location of the Iban corpus"
    exit 1
fi
if [ ! -d "$corpus" ] ; then
    echo >&2 "The directory $corpus does not exist"
fi


echo "Preparing train and test data"
mkdir -p data data/local data/train data/dev

for x in train dev; do
    echo "Copy spk2utt, utt2spk, wav.scp, text for $x"
    cp $corpus/data/$x/text     data/$x/text    || exit 1;
    cp $corpus/data/$x/spk2utt  data/$x/spk2utt || exit 1;
    cp $corpus/data/$x/utt2spk  data/$x/utt2spk || exit 1;

    # the corpus wav.scp contains physical paths, so we just re-generate
    # the file again from scratchn instead of figuring out how to edit it
    for rec in $(awk '{print $1}' $corpus/data/$x/text) ; do
        spk=${rec%_*}
        filename=$corpus/data/wav/$spk/${rec}.wav
        if [ ! -f "$filename" ] ; then
            echo >&2 "The file $filename could not be found ($rec)"
            exit 1
        fi
        # we might want to store physical paths as a general rule
        filename=$(utils/make_absolute.sh $filename)
        echo "$rec $filename"
    done > data/$x/wav.scp

    # fix_data_dir.sh fixes common mistakes (unsorted entries in wav.scp,
    # duplicate entries and so on). Also, it regenerates the spk2utt from
    # utt2sp
    utils/fix_data_dir.sh data/$x
done

echo "Copying language model"
if [ -f $corpus/LM/iban-lm-o3.arpa.tar.gz ] ; then
    tar zxf $corpus/LM/iban-lm-o3.arpa.tar.gz -C data/local/
fi

echo "Data preparation completed."

