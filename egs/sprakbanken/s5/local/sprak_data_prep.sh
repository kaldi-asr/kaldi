#!/bin/bash

# Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Copyright 2013-2014  Mirsk Digital Aps (Author: Andreas Kirkedal)
# Copyright 2015-2016  Andreas Kirkedal
# Apache 2.0.


dir=`pwd`/data/local/data
lmdir=`pwd`/data/local/transcript_lm
traindir=`pwd`/data/local/trainsrc
testdir=`pwd`/data/local/testsrc
devdir=`pwd`/data/local/devsrc
rm -rf $lmdir $traindir $testdir $devdir
mkdir -p $dir $lmdir $traindir $testdir $devdir
local=`pwd`/local
utils=`pwd`/utils

. ./path.sh

# Checks if python3 is available on the system and install python3 in userspace if not
# This recipe currently relies on version 3 because python3 uses utf8 as internal 
# string representation

#if ! which python3 >&/dev/null; then
#  echo "Installing python3 since not on your path."
#  pushd $KALDI_ROOT/tools || exit 1;
#  extras/install_python3.sh || exit 1;
#  popd
#fi

if [ ! -d $dir/download ]; then
    mkdir -p $dir/download/0565-1 $dir/download/0565-2
fi 

echo "Downloading and unpacking sprakbanken to $dir/corpus_processed. This will take a while."

if [ ! -f $dir/download/da.16kHz.0565-1.tar.gz ]; then 
    ( wget http://www.nb.no/sbfil/talegjenkjenning/16kHz/da.16kHz.0565-1.tar.gz --directory-prefix=$dir/download )
fi

if [ ! -f $dir/download/da.16kHz.0565-2.tar.gz ]; then 
    ( wget http://www.nb.no/sbfil/talegjenkjenning/16kHz/da.16kHz.0565-2.tar.gz --directory-prefix=$dir/download )
fi

if [ ! -f $dir/download/da.16kHz.0611.tar.gz ]; then 
    ( wget http://www.nb.no/sbfil/talegjenkjenning/16kHz/da.16kHz.0611.tar.gz --directory-prefix=$dir/download )
fi    
wait

echo "Corpus files downloaded."

if [ ! -d $dir/download/0611 ]; then
    echo "Unpacking files."
    tar -xzf $dir/download/da.16kHz.0565-1.tar.gz -C $dir/download/0565-1
    tar -xzf $dir/download/da.16kHz.0565-2.tar.gz -C $dir/download/0565-2 
    tar -xzf $dir/download/da.16kHz.0611.tar.gz -C $dir/download    

    # Note: rename "da 0611 test" to "da_0611_test" for this to work
    mv $dir/download/"da 0611 test" $dir/download/0611
    wait     
    echo "Corpus unpacked succesfully."
fi


sph2pipe=$(which sph2pipe) || sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
   exit 1;
fi


echo "Converting downloaded files to a format consumable by Kaldi scripts."

rm -rf $dir/corpus_processed 
mkdir -p $dir/corpus_processed/training/0565-1 $dir/corpus_processed/training/0565-2 $dir/corpus_processed/training/0611_Stasjon05


# Create parallel file lists and text files, but keep sound files in the same location to save disk space
# Writes the lists to data/local/data (~ 310h)
echo "Creating parallel data for training data."
python $local/sprak2kaldi.py $dir/download/0565-1 $dir/corpus_processed/training/0565-1   # ~130h
python $local/sprak2kaldi.py $dir/download/0565-2 $dir/corpus_processed/training/0565-2   # ~115h
python $local/sprak2kaldi.py $dir/download/0611/Stasjon05 $dir/corpus_processed/training/0611_Stasjon05  # ~51h 

(
# Ditto dev set (~ 16h)
    echo "Creating parallel data for test data."
    rm -rf $dir/corpus_processed/dev03 
    mkdir -p $dir/corpus_processed/dev03 
    python $local/sprak2kaldi.py $dir/download/0611/Stasjon03 $dir/corpus_processed/dev03 || exit 1;
)

(
# Ditto test set (about 9 hours)
    echo "Creating parallel data for development data."
    rm -rf $dir/corpus_processed/test06 
    mkdir -p $dir/corpus_processed/test06 
    python $local/sprak2kaldi.py $dir/download/0611/Stasjon06 $dir/corpus_processed/test06 || exit 1;
)

# Create the LM training data 
# Test and dev data is disjoint from training data, so we use those transcripts)

# Because training data is read aloud, there are many occurences of the same
# sentence and bias towards the domain. Make a version where  
# the sentences are unique to reduce bias.

(
    echo "Writing the LM text to file and normalising."
    cat $dir/corpus_processed/training/0565-1/txtlist $dir/corpus_processed/training/0565-2/txtlist | while read l; do cat $l; done > $lmdir/lmsents
    python local/normalize_transcript.py local/norm_dk/numbersLow.tbl $lmdir/lmsents $lmdir/lmsents.norm
    local/norm_dk/format_text.sh lm $lmdir/lmsents.norm > $lmdir/transcripts.txt
    sort -u $lmdir/transcripts.txt > $lmdir/transcripts.uniq
)

# Combine training file lists
echo "Combine file lists."
cat $dir/corpus_processed/training/0565-1/txtlist $dir/corpus_processed/training/0565-2/txtlist $dir/corpus_processed/training/0611_Stasjon05/txtlist > $dir/traintxtfiles
cat $dir/corpus_processed/training/0565-1/sndlist $dir/corpus_processed/training/0565-2/sndlist $dir/corpus_processed/training/0611_Stasjon05/sndlist > $dir/trainsndfiles

# Move test file lists to the right location
cp $dir/corpus_processed/dev03/txtlist $dir/devtxtfiles
cp $dir/corpus_processed/dev03/sndlist $dir/devsndfiles

# Move test file lists to the right location
cp $dir/corpus_processed/test06/txtlist $dir/testtxtfiles
cp $dir/corpus_processed/test06/sndlist $dir/testsndfiles

# Write wav.scp, utt2spk and text.unnormalised for train, test and dev sets with
# Use sph2pipe because the wav files are actually sph files
echo "Creating wav.scp, utt2spk and text.unnormalised for train, test and dev" 
python $local/data_prep.py $dir/traintxtfiles $traindir $dir/trainsndfiles $sph2pipe 
python $local/data_prep.py $dir/testtxtfiles $testdir $dir/testsndfiles $sph2pipe 
python $local/data_prep.py $dir/devtxtfiles $devdir $dir/devsndfiles $sph2pipe 


# Create the main data sets
local/create_datasets.sh $testdir data/test 
local/create_datasets.sh $devdir data/dev 
local/create_datasets.sh $traindir data/train 

## TODO

# Extract gender from spl files 
# Decide how to handle cases with no gender specification

echo "Data preparation succeeded"
