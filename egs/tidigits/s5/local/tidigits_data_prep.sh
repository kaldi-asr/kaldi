#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

. ./path.sh # Needed for KALDI_ROOT

if [ $# -ne 1 ]; then
   echo "Argument should be the TIDIGITS directory, see ../run.sh for example."
   exit 1;
fi

tidigits=$1

tmpdir=`pwd`/data/local/data
mkdir -p $tmpdir

# Note: the .wav files are not in .wav format but "sphere" format (this was 
# produced in the days before Windows).

if [ -d $tidigits/data ]; then
  rootdir=$tidigits/data/adults
elif [ -d $tidigits/tidigits ]; then 
  # This is, I think, a modified
  # version of the format that exists at BUT.
  rootdir=$tidigits/tidigits
else
  echo "Tidigits directory $tidigits does not have expected format."
fi

find $rootdir/train -name '*.wav' > $tmpdir/train.flist
n=`cat $tmpdir/train.flist | wc -l`
[ $n -eq 8623 ] || echo Unexpected number of training files $n versus 8623

find $rootdir/test -name '*.wav' > $tmpdir/test.flist
n=`cat $tmpdir/test.flist | wc -l`
[ $n -eq 8700 ] || echo Unexpected number of test files $n versus 8700


sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
   exit 1;
fi

for x in train test; do
  # get scp file that has utterance-ids and maps to the sphere file.
  cat $tmpdir/$x.flist | perl -ane 'm|/(..)/([1-9zo]+[ab])\.wav| || die "bad line $_"; print "$1_$2 $_"; ' \
   | sort > $tmpdir/${x}_sph.scp
  # turn it into one that has a valid .wav format in the modern sense (i.e. RIFF format, not sphere).
  # This file goes into its final location
  mkdir -p data/$x
  awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < $tmpdir/${x}_sph.scp > data/$x/wav.scp

  # Now get the "text" file that says what the transcription is.
  cat data/$x/wav.scp | 
   perl -ane 'm/^(.._([1-9zo]+)[ab]) / || die; $text = join(" ", split("", $2)); print "$1 $text\n";' \
    <data/$x/wav.scp >data/$x/text

  # now get the "utt2spk" file that says, for each utterance, the speaker name.  
  perl -ane 'm/^((..)_\S+) / || die; print "$1 $2\n"; ' \
    <data/$x/wav.scp >data/$x/utt2spk
  # create the file that maps from speaker to utterance-list.
  utils/utt2spk_to_spk2utt.pl <data/$x/utt2spk >data/$x/spk2utt
done

echo "Data preparation succeeded"
