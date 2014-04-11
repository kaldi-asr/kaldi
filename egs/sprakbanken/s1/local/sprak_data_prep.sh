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



#cd $dir

#mkdir -p $4/parallel/training $4/parallel/test

## Create a parallel corpus with text/wav pairs with unique identifiers using session, speaker and utterance ids
## The input text files are in iso8859-1, the output text files are in utf-8 and the wav files are actually sph

## Training part
#python3 $local/sprak2parallel.py $1 $4/parallel/training/$(basename $1) &
#python3 $local/sprak2parallel.py $2 $4/parallel/training/$(basename $2) &

## Testing part
#python3 $local/sprak2parallel.py $3 $4/parallel/test/$(basename $3) &

#wait

## Creates a list of all the text files in the corpus
#find $4/parallel/training -type f -name \*.txt > traintxtfiles
#find $4/parallel/test -type f -name \*.txt > testtxtfiles

## Creates an intermediate "text" file and the wav.scp and the utt2spk files
## The wav.scp file contains commands to sph2pipe instead of filenames
#python3 $local/sprak_data_prep.py traintxtfiles $traindir &

## Ditto test corpus
#python3 $local/sprak_data_prep.py testtxtfiles $testdir &

#wait
 
#mkdir -p parallel/training parallel/test06 parallel/dev03

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

# IF SPEAKER INFO CAN BE INCORPORATED
#if [ ! -f wsj0-train-spkrinfo.txt ]; then
#  echo "Could not get the spkrinfo.txt file from LDC website (moved)?"
#  echo "This is possibly omitted from the training disks; couldn't find it." 
#  echo "Everything else may have worked; we just may be missing gender info"
#  echo "which is only needed for VTLN-related diagnostics anyway."
#  exit 1
#fi
# Note: wsj0-train-spkrinfo.txt doesn't seem to be on the disks but the
# LDC put it on the web.  Perhaps it was accidentally omitted from the
# disks.  

#cat links/11-13.1/wsj0/doc/spkrinfo.txt \
#    links/13-32.1/wsj1/doc/evl_spok/spkrinfo.txt \
#    links/13-34.1/wsj1/doc/dev_spok/spkrinfo.txt \
#    links/13-34.1/wsj1/doc/train/spkrinfo.txt \
#   ./wsj0-train-spkrinfo.txt  | \
#    perl -ane 'tr/A-Z/a-z/; m/^;/ || print;' | \
#   awk '{print $1, $2}' | grep -v -- -- | sort | uniq > spk2gender


echo "Data preparation succeeded"
