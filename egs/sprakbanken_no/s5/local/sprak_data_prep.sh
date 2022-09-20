#!/usr/bin/env bash

# Copyright 2010-2012 Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Copyright 2014 Mirsk Digital ApS  (Author: Andreas Kirkedal)
# Copyright 2022  Institute of Language and Speech Processing (ILSP), AthenaRC (Author: Thodoris Kouzelis)
# Apache 2.0.

dir=`pwd`/data/local/data
lmdir=`pwd`/data/local/transcript_lm
train=$dir/download/metadata/train
test_dev=$dir/download/metadata/test_dev
dev=$dir/download/metadata/dev
test=$dir/download/metadata/test
rm -rf $lmdir 
mkdir -p $dir $lmdir 
local=`pwd`/local
utils=`pwd`/utils

# channel=1: Close channel
# channel=2: Distant chennel
# channel=begge: Both channels
# The results shown in the RESULT file where produced using audio from channel 1 (close).
# For more info read: https://www.nb.no/sbfil/talegjenkjenning/16kHz_2020/no_2020/no-16khz_reorganized_english.pdf 
channel=1 

. ./path.sh

if [ ! -d $dir/download ]; then
    mkdir -p $dir/download/
fi 


echo "Downloading and unpacking sprakbanken to $dir/corpus_processed. This will take a while."

if [ ! -f $dir/download/lydfiler_16_1_a.tar.gz ]; then 
   ( wget --tries 100 https://www.nb.no/sbfil/talegjenkjenning/16kHz_2020/no_2020/lydfiler_16_${channel}_a.tar.gz --directory-prefix=$dir/download )
fi

if [ ! -f $dir/download/lydfiler_16_1_b.tar.gz ]; then 
   ( wget --tries 100 https://www.nb.no/sbfil/talegjenkjenning/16kHz_2020/no_2020/lydfiler_16_${channel}_b.tar.gz --directory-prefix=$dir/download )
fi

if [ ! -f $dir/download/lydfiler_16_1_c.tar.gz ]; then 
   ( wget --tries 100 https://www.nb.no/sbfil/talegjenkjenning/16kHz_2020/no_2020/lydfiler_16_${channel}_c.tar.gz --directory-prefix=$dir/download )
fi

if [ ! -f $dir/download/lydfiler_16_1_d.tar.gz ]; then 
   ( wget --tries 100 https://www.nb.no/sbfil/talegjenkjenning/16kHz_2020/no_2020/lydfiler_16_${channel}_d.tar.gz --directory-prefix=$dir/download )
fi

if [ ! -f $dir/download/ADB_NOR_0463.tar.gz ]; then
   ( wget --tries 100 https://www.nb.no/sbfil/talegjenkjenning/16kHz_2020/no_2020/ADB_NOR_0463.tar.gz --directory-prefix=$dir/download )
fi

if [ ! -f $dir/download/ADB_NOR_0464.tar.gz ]; then
   ( wget --tries 100 https://www.nb.no/sbfil/talegjenkjenning/16kHz_2020/no_2020/ADB_NOR_0464.tar.gz --directory-prefix=$dir/download )
fi

if [ ! -f $dir/download/ADB_OD_Nor.NOR.tar.gz ]; then
    ( wget --tries 100 https://www.nb.no/sbfil/talegjenkjenning/16kHz_2020/no_2020/ADB_OD_Nor.NOR.tar.gz --directory-prefix=$dir/download )
fi

if [ ! -f $dir/download/metadata_no_csv.zip ]; then
    ( wget --tries 100 https://www.nb.no/sbfil/talegjenkjenning/16kHz_2020/no_2020/metadata_no_csv.zip --directory-prefix=$dir/download )
fi



echo "Corpus files downloaded."
if [ ! -d $dir/download/no ]; then
   tar -xzf $dir/download/lydfiler_16_1_a.tar.gz -C $dir/download
   tar -xzf $dir/download/lydfiler_16_1_b.tar.gz -C $dir/download
   tar -xzf $dir/download/lydfiler_16_1_c.tar.gz -C $dir/download
   tar -xzf $dir/download/lydfiler_16_1_d.tar.gz -C $dir/download
fi

if [ ! -d $dir/download/metadata ]; then 

    mkdir -p $train $test_dev $test $dev
    tar -xzf $dir/download/ADB_NOR_0463.tar.gz -C $train
    tar -xzf $dir/download/ADB_NOR_0464.tar.gz -C $test_dev
fi

# Splits test_dev to test and dev.
# Use 10 recordings for the test and dev set respectivly, the rest are merged with the training set.
echo "Create test and dev splits"
ls $test_dev | head -n 10 | xargs -i mv $test_dev/{} $test
ls $test_dev | head -n 10 | xargs -i mv $test_dev/{} $dev
mv $test_dev/* $train
rm -rf $test_dev

# Write wav.scp, utt2spk, spk2utt, spk2gender and text for train, test and dev.
echo "Creating wav.scp, utt2spk and text for train, test and dev"
for dataset in train test dev; do
    mkdir data/$dataset
    python3 local/data_prep.py $dir/download/metadata/$dataset $dir/download/no/ data/$dataset
    utils/fix_data_dir.sh data/$dataset
    utils/validate_data_dir.sh --no-feats data/$dataset || exit 1;
done

# Create the LM training data.
echo "Writing the LM text to file"
awk '{$1="";print $0}' data/train/text | sort -u > $lmdir/transcripts.uniq

