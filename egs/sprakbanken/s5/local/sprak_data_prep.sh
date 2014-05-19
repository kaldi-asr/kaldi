#!/bin/bash

# Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Copyright 2013-2014  Mirsk Digital Aps (Author: Andreas Kirkedal)
# Apache 2.0.


dir=`pwd`/data/local/data
lmdir=`pwd`/data/local/arpa_lm
traindir=`pwd`/data/train
testdir=`pwd`/data/test
devdir=`pwd`/data/dev
mkdir -p $dir $lmdir $traindir $testdir $devdir
local=`pwd`/local
utils=`pwd`/utils


# Checks if python3 is available on the system and install python3 in userspace if not
# This recipe currently relies on version 3 because python3 uses utf8 as internal 
# representation string representation

$KALDI_ROOT/extras/install_python3.sh

if [ ! -d $dir/corpus ]; then

    mkdir -p $dir/corpus/0565-1 $dir/corpus/0565-2
fi 

echo "Downloading and unpacking sprakbanken to $dir/corpus. This will take a while."

if [ ! -f $dir/corpus/da.16kHz.0565-1.tar.gz ]; then 
    ( wget http://www.nb.no/sbfil/talegjenkjenning/16kHz/da.16kHz.0565-1.tar.gz --directory-prefix=$dir/corpus ) &
fi

if [ ! -f $dir/corpus/da.16kHz.0565-2.tar.gz ]; then 
    ( wget http://www.nb.no/sbfil/talegjenkjenning/16kHz/da.16kHz.0565-2.tar.gz --directory-prefix=$dir/corpus ) &
fi

if [ ! -f $dir/corpus/da.16kHz.0565-1.tar.gz ]; then 
    ( wget http://www.nb.no/sbfil/talegjenkjenning/16kHz/da.16kHz.0611.tar.gz --directory-prefix=$dir/corpus ) &
fi    
wait

echo "Corpus files downloaded."

if [ ! -d $dir/corpus/0611 ]; then
    echo "Unpacking files."
    tar -xzf $dir/corpus/da.16kHz.0565-1.tar.gz -C $dir/corpus/0565-1 &
    tar -xzf $dir/corpus/da.16kHz.0565-2.tar.gz -C $dir/corpus/0565-2 &
    tar -xzf $dir/corpus/da.16kHz.0611.tar.gz -C $dir/corpus    

    # Note: rename "da 0611 test" to "da_0611_test" for this to work
    mv $dir/corpus/"da 0611 test" $dir/corpus/0611
    wait     
    echo "Corpus unpacked succesfully."
fi

. ./path.sh # Needed for KALDI_ROOT
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
   exit 1;
fi


echo "Converting corpus files to a format consumable by Kaldi scripts."

mkdir -p $dir/corpus/training/0565-1 $dir/corpus/training/0565-2


# Create parallel file lists and text files, but keep sound files in the same location
# Writes the lists to data/local/data (~ 310h)
python3 $local/sprak2kaldi.py $dir/corpus/0565-1 $dir/corpus/training/0565-1 & # ~130h
python3 $local/sprak2kaldi.py $dir/corpus/0565-2 $dir/corpus/training/0565-2 & # ~115h
python3 $local/sprak2kaldi.py $dir/corpus/0611/Stasjon05 $dir/corpus/training/0611_Stasjon05 & # ~51h

# Ditto dev set (~ 16h)
python3 $local/sprak2kaldi.py $dir/corpus/0611/Stasjon03 $dir/corpus/dev03 & 

# Ditto test set (about 9 hours)
python3 $local/sprak2kaldi.py $dir/corpus/0611/Stasjon06 $dir/corpus/test06 &

wait

# Combine training file lists
cat $dir/corpus/training/0565-1/txtlist $dir/corpus/training/0565-2/txtlist $dir/corpus/training/0611_Stasjon05/txtlist > $dir/traintxtfiles
cat $dir/corpus/training/0565-1/sndlist $dir/corpus/training/0565-2/sndlist $dir/corpus/training/0611_Stasjon05/sndlist > $dir/trainsndfiles

# LM training files (test data is disjoint from training data)
cat $dir/corpus/training/0565-1/txtlist $dir/corpus/training/0565-2/txtlist > $dir/lmtxtfiles

# Move test file lists to the right location
mv $dir/corpus/dev03/txtlist $dir/devtxtfiles
mv $dir/corpus/dev03/sndlist $dir/devsndfiles


# Move test file lists to the right location
mv $dir/corpus/test06/txtlist $dir/testtxtfiles
mv $dir/corpus/test06/sndlist $dir/testsndfiles

python3 $local/sprak_data_prep.py $dir/traintxtfiles $traindir $dir/trainsndfiles $sph2pipe &
python3 $local/sprak_data_prep.py $dir/testtxtfiles $testdir $dir/testsndfiles $sph2pipe &
python3 $local/sprak_data_prep.py $dir/devtxtfiles $devdir $dir/devsndfiles $sph2pipe &

cat $dir/lmtxtfiles | while read l; do cat $l; done > $dir/lmsents

wait

## TODO

# Extract gender from spl files 
# Decide how to handle cases with no gender specification

echo "Data preparation succeeded"
