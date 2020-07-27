#!/usr/bin/env bash


# Copyright 2014  University of Edinburgh (Author: Pawel Swietojanski)
#           2016  Johns Hopkins University (Author: Daniel Povey)
# Copyright 2019  IBM Corp. (Author: Michael Picheny) Adapted AMI recipe to MALACH corpus
# MALACH Corpus training data preparation
# Apache 2.0

# Note: this is called by ../run.sh.

. ./path.sh

#check existing directories
if [ $# != 2 ]; then
  echo "Usage: $0 /path/to/MALACH"
  echo "e.g. $0 /foo/bar/MALACH"
  exit 1;
fi

MALACH_DIR=$1
SET=$2
SEGS=data/local/annotations/$SET.txt

dir=data/local/$SET
odir=data/${SET}_orig
mkdir -p $dir

cp $MALACH_DIR/$SET/* $dir


# Audio data directory check
if [ ! -d $MALACH_DIR ]; then
  echo "Error: run.sh requires a directory argument"
  exit 1;
fi

# And transcripts check
if [ ! -f $SEGS ]; then
  echo "Error: File $SEGS no found (run malach_text_prep.sh)."
  exit 1;
fi


sort -k 2 $dir/utt2spk | utils/utt2spk_to_spk2utt.pl > $dir/spk2utt || exit 1;

# Copy stuff into its final locations
mkdir -p $odir
for f in spk2utt utt2spk wav.scp text segments reco2file_and_channel; do
  cp $dir/$f $odir/$f || exit 1;
done

#Produce STMs for sclite scoring
local/convert2stm.pl $dir > $odir/stm
cp $MALACH_DIR/glm $odir/glm

utils/validate_data_dir.sh --no-feats $odir || exit 1;

echo MALACH $SET set data preparation succeeded.

