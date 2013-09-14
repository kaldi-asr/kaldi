#!/bin/bash

# Copyright 2013   (Author: Bagher BabaAli)
# Apache 2.0.


if [ $# -ne 1 ]; then
   echo "Argument should be the Timit directory, see ../run.sh for example."
   exit 1;
fi


dir=`pwd`/data/local/data
mkdir -p $dir
local=`pwd`/local
utils=`pwd`/utils
conf=`pwd`/conf

. ./path.sh # Needed for KALDI_ROOT

sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
   exit 1;
fi

cd $dir

# Make directory of links to the TIMIT disk.  This relies on the command
# line arguments being absolute pathnames.
rm -r links/ 2>/dev/null
mkdir links/

ln -s $* links

# Do some basic checks that we have what we expected.
if [ ! -d $*/TRAIN -o ! -d $*/TEST ]; then
  echo "timit_data_prep.sh: Spot check of command line argument failed"
  echo "Command line argument must be absolute pathname to TIMIT directory"
  echo "with name like /export/corpora5/LDC/LDC93S1/timit/TIMIT"
  exit 1;
fi

# This version for TRAIN

TrainDir=$*/TRAIN
find -L $TrainDir \( -iname '*.WAV' -o -iname '*.wav' \) > train.flist
nl=`cat train.flist | wc -l`
[ "$nl" -eq 4620 ] || echo "Warning: expected 4620 lines in train.flist, got $nl"

# Now for the TEST.

TestDir=$*/TEST
find -L $TestDir \( -iname '*.WAV' -o -iname '*.wav' \) > test.flist

nl=`cat test.flist | wc -l`
[ "$nl" -eq 1680 ] || echo "Warning: expected 1680 lines in test.flist, got $nl"


# Finding the transcript files:
find -L $TrainDir \( -iname '*.PHN' -o -iname '*.phn' \) > train_phn.flist
find -L $TestDir \( -iname '*.PHN' -o -iname '*.phn' \)  > test_phn.flist

# Convert the transcripts into our format (no normalization yet)
for x in train test; do
   $local/timit_flist2scp.pl $x.flist | sort > ${x}_sph.scp
   cat ${x}_sph.scp | awk '{print $1}' > ${x}.uttids
   cat ${x}.uttids | $local/timit_find_transcripts.pl  ${x}_phn.flist > ${x}_phn.trans
done

# Do normalization steps. 
cat train_phn.trans | $local/timit_norm_trans.pl -i - -m $conf/phones.60-48-39.map  -to 48 | sort > train.txt || exit 1;


for x in test; do
   cat ${x}_phn.trans | $local/timit_norm_trans.pl -i - -m $conf/phones.60-48-39.map -to 39 | sort > $x.txt || exit 1;
done

# Create scp's with wav's.
for x in train test; do
  awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < ${x}_sph.scp > ${x}_wav.scp
done

# Make the utt2spk and spk2utt files.
for x in train test; do
    cut -f1 -d'_'  $x.uttids | paste -d' ' $x.uttids - > $x.utt2spk 
   cat $x.utt2spk | $utils/utt2spk_to_spk2utt.pl > $x.spk2utt || exit 1;
done

# Make the spk2gender files.
for x in train test; do
   cat $x.spk2utt | awk '{print $1}' | perl -ane 'chop; m:^.:; print "$_ $&\n";' > $x.spk2gender
done



echo "Data preparation succeeded"
