#!/usr/bin/env bash

# Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Copyright 2013-2014  Mirsk Digital Aps (Author: Andreas Kirkedal)
# Copyright 2016 KTH Royal Institute of Technology (Author: Emelie Kullmann)
# Apache 2.0.


dir=`pwd`/data/local/data
lmdir=`pwd`/data/local/transcript_lm
traindir=`pwd`/data/local/trainsrc
testdir=`pwd`/data/local/testsrc
rm -rf $lmdir $traindir $testdir $devdir
mkdir -p $dir $lmdir $traindir $testdir $devdir
local=`pwd`/local
utils=`pwd`/utils


. ./path.sh

# Checks if python3 is available on the system and install python3 in userspace if not
# This recipe currently relies on version 3 because python3 uses utf8 as internal 
# string representation

#if ! which python3 >&/dev/null; then
#  echo "Python3 is not installed, to install it you should probably do:"
#  echo "sudo apt-get install python3" || exit 1;
#fi

if [ ! -d $dir/download ]; then
    mkdir -p $dir/download/0467-1 $dir/download/0467-2 $dir/download/0467-3
fi 

echo "Downloading and unpacking sprakbanken to $dir/corpus_processed. This will take a while."

if [ ! -f $dir/download/sve.16khz.0467-1.tar.gz ]; then 
    ( wget --tries 100 http://www.nb.no/sbfil/talegjenkjenning/16kHz/sve.16khz.0467-1.tar.gz --directory-prefix=$dir/download )
fi

if [ ! -f $dir/download/sve.16khz.0467-2.tar.gz ]; then 
    ( wget --tries 100 http://www.nb.no/sbfil/talegjenkjenning/16kHz/sve.16khz.0467-2.tar.gz --directory-prefix=$dir/download )
fi

if [ ! -f $dir/download/sve.16khz.0467-3.tar.gz ]; then 
    ( wget --tries 100 http://www.nb.no/sbfil/talegjenkjenning/16kHz/sve.16khz.0467-3.tar.gz --directory-prefix=$dir/download )
fi

if [ ! -f $dir/download/sve.16khz.0467-1.tar.gz ]; then 
    ( wget --tries 100 http://www.nb.no/sbfil/talegjenkjenning/16kHz/sve.16khz.0468.tar.gz --directory-prefix=$dir/download )
fi    

echo "Corpus files downloaded."

if [ ! -d $dir/download/0468 ]; then
    echo "Unpacking files."
    tar -xzf $dir/download/sve.16khz.0467-1.tar.gz -C $dir/download/0467-1
    tar -xzf $dir/download/sve.16khz.0467-2.tar.gz -C $dir/download/0467-2
    tar -xzf $dir/download/sve.16khz.0467-3.tar.gz -C $dir/download/0467-3
    tar -xzf $dir/download/sve.16khz.0468.tar.gz -C $dir/download/0468    

     
    echo "Corpus unpacked succesfully."
fi

sph2pipe=$(which sph2pipe) || sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe . Did you run 'make' in the tools directory?";
   exit 1;
fi

echo "done"

echo "Converting downloaded files to a format consumable by Kaldi scripts."

rm -rf $dir/corpus_processed 
mkdir -p $dir/corpus_processed/training/0467-1 $dir/corpus_processed/training/0467-2 $dir/corpus_processed/training/0467-3 

# Create parallel file lists and text files, but keep sound files in the same location to save disk space
# Writes the lists to data/local/data (~ 310h)
echo "Creating parallel data for training data."
python $local/sprak2kaldi.py $dir/download/0467-1 $dir/corpus_processed/training/0467-1  # ~140h
python $local/sprak2kaldi.py $dir/download/0467-2 $dir/corpus_processed/training/0467-2  # ~125h
python $local/sprak2kaldi.py $dir/download/0467-3 $dir/corpus_processed/training/0467-3  # ~128h

mv $dir/corpus_processed/training/0467-1/'r4670118.791213 8232' $dir/corpus_processed/training/0467-1/'r4670118.791213_8232'
for f in $dir/corpus_processed/training/0467-1/r4670118.791213_8232/*.txt; do
    mv "$f" "${f// /_}";
done

(
# Ditto test set (~ 93h)
    echo "Creating parallel data for test data."
    rm -rf $dir/corpus_processed/test/0468 
    mkdir -p $dir/corpus_processed/test/0468 
    python $local/sprak2kaldi.py $dir/download/0468 $dir/corpus_processed/test/0468
) 


# Create the LM training data 
(
    echo "Writing the LM text to file and normalising."
    cat $dir/corpus_processed/training/0467-1/txtlist $dir/corpus_processed/training/0467-2/txtlist $dir/corpus_processed/training/0467-3/txtlist | while read l; do cat $l; done > $lmdir/lmsents
    python local/normalize_transcript.py $lmdir/lmsents $lmdir/lmsents.norm
    sort -u $lmdir/lmsents.norm > $lmdir/transcripts.uniq
)

# Combine training file lists
echo "Combine file lists."
cat $dir/corpus_processed/training/0467-1/txtlist $dir/corpus_processed/training/0467-2/txtlist $dir/corpus_processed/training/0467-3/txtlist > $dir/traintxtfiles
cat $dir/corpus_processed/training/0467-1/sndlist $dir/corpus_processed/training/0467-2/sndlist $dir/corpus_processed/training/0467-3/sndlist > $dir/trainsndfiles


# Move test file lists to the right location
cp $dir/corpus_processed/test/0468/txtlist $dir/testtxtfiles
cp $dir/corpus_processed/test/0468/sndlist $dir/testsndfiles

# Write wav.scp, utt2spk and text.unnormalised for train, test and dev sets with
# Use sph2pipe because the wav files are actually sph files
echo "Creating wav.scp, utt2spk and text.unnormalised for train, test and dev" 
python3 $local/data_prep.py $dir/traintxtfiles $traindir $dir/trainsndfiles $sph2pipe
python3 $local/data_prep.py $dir/testtxtfiles $testdir $dir/testsndfiles $sph2pipe



# Create the main data sets
local/create_datasets.sh $testdir data/test 
local/create_datasets.sh $traindir data/train 




echo "Data preparation succeeded"
