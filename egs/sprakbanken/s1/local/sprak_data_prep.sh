#!/bin/bash

# Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Copyright 2013-2014  Mirsk Digital Aps (Author: Andreas Kirkedal)
# Apache 2.0.


if [ $# -le 3 ]; then
   echo "Arguments should be a list of directories and a destination, see ../run.sh for example."
   exit 1;
fi

#rm -rf data 

dir=`pwd`/data/local/data
lmdir=`pwd`/data/local/arpa_lm
traindir=`pwd`/data/train
testdir=`pwd`/data/test
devdir=`pwd`/data/dev
mkdir -p $dir $lmdir $traindir $testdir $devdir
local=`pwd`/local
utils=`pwd`/utils



. ./path.sh # Needed for KALDI_ROOT
export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
   exit 1;
fi


# Create parallel file lists and text files, but keep sound files in the same location
# Note: rename "da 0611 test" to "da_0611_test" for this to work
find $3 -name "* *" -type d | rename 's/ /_/g'
# Writes the lists to data/local/data (~ 310h)
python3 $local/sprak2kaldi.py $1 $dir/parallel/training/$(basename $1) & # ~130h
python3 $local/sprak2kaldi.py $2 $dir/parallel/training/$(basename $2) & # ~115h
python3 $local/sprak2kaldi.py $3/Stasjon05 $dir/parallel/training/0611_Stasjon05 & # ~51h

# Ditto dev set (~ 16h)
python3 $local/sprak2kaldi.py $3/Stasjon03 $dir/parallel/dev03 & 

# Ditto test set (about 9 hours)
python3 $local/sprak2kaldi.py $3/Stasjon06 $dir/parallel/test06 &

wait

# Combine training file lists
cat $dir/parallel/training/$(basename $1)/txtlist $dir/parallel/training/$(basename $2)/txtlist $dir/parallel/training/0611_Stasjon05/txtlist > $dir/traintxtfiles
cat $dir/parallel/training/$(basename $1)/sndlist $dir/parallel/training/$(basename $2)/sndlist $dir/parallel/training/0611_Stasjon05/sndlist > $dir/trainsndfiles

# LM training files (test data is disjoint from training data)
cat $dir/parallel/training/$(basename $1)/txtlist $dir/parallel/training/$(basename $2)/txtlist > $dir/lmtxtfiles

# Move test file lists to the right location
mv $dir/parallel/dev03/txtlist $dir/devtxtfiles
mv $dir/parallel/dev03/sndlist $dir/devsndfiles


# Move test file lists to the right location
mv $dir/parallel/test06/txtlist $dir/testtxtfiles
mv $dir/parallel/test06/sndlist $dir/testsndfiles


python3 $local/sprak_data_prep.py $dir/traintxtfiles $traindir $dir/trainsndfiles $sph2pipe &
python3 $local/sprak_data_prep.py $dir/testtxtfiles $testdir $dir/testsndfiles $sph2pipe &
python3 $local/sprak_data_prep.py $dir/devtxtfiles $devdir $dir/devsndfiles $sph2pipe &
sprak_prep_lm.sh $dir/lmtxtfiles $dir/trainsents
wait

## TODO

# Extract gender from spl files 
# Decide how to handle cases with no gender specification

echo "Data preparation succeeded"
