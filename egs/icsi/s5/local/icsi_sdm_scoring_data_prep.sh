#!/bin/bash

# Copyright 2014 University of Edinburgh (Author: Pawel Swietojanski)
#           2018 Emotech LTD (Author: Pawel Swietojanski)
# ICSI Corpus training data preparation
# Apache 2.0
# To be run from one directory above this script.

. path.sh

if [ $# != 4 ]; then
  echo "Usage: icsi_sdm_scoring_data_prep_edin.sh /path/to/AMI seg-file set-name mic"
  exit 1; 
fi 

CORPUS_DIR=$1
SEGS=$2 #assuming here all normalisation stuff was done
SET=$3
MICNUM=$4
mic="m$MICNUM"

tmpdir=data/local/sdm/$mic/$SET
dir=data/sdm/$mic/$SET
channels=data/local/channels.bmf

mkdir -p $tmpdir

if [ ! -f $channels ]; then
  echo 'Meeting to channel mapping file missing'
  exit 1;
fi

# Audio data directory check
if [ ! -d $CORPUS_DIR ]; then
  echo "Error: run.sh requires a directory argument"
  exit 1;
fi

cut -d" " -f1,$(($MICNUM+1)) $channels | awk '{print $1".*"$2}' > $tmpdir/channels_regex
# list all wav file you can
find $CORPUS_DIR -iname "*.wav" | sort > $tmpdir/wav.flist.all
# and keep only these we want
grep -f $tmpdir/channels_regex $tmpdir/wav.flist.all > $tmpdir/wav.flist

n=`cat $tmpdir/wav.flist | wc -l`
echo "In total, $n files were found."

# (1a) Transcriptions preparation
# here we start with normalised transcripts

awk '{meeting=$1; channel="SDM"; speaker=$3; stime=$4; etime=$5;
 printf("ICSI_%s_%s_%s_%07.0f_%07.0f", meeting, channel, speaker, int(100*stime+0.5), int(100*etime+0.5));
 for(i=6;i<=NF;i++) printf(" %s", $i); printf "\n"}' $SEGS | sort  > $tmpdir/text

# (1c) Make segment files from transcript
#segments file format is: utt-id side-id start-time end-time, e.g.:
#AMI_ES2011a_H00_FEE041_0003415_0003484
awk '{ 
       segment=$1;
       split(segment,S,"[_]");
       audioname=S[1]"_"S[2]"_"S[3]; startf=S[5]; endf=S[6];
       print segment " " audioname " " startf/100 " " endf/100 " " 0
}' < $tmpdir/text > $tmpdir/segments

cat $tmpdir/wav.flist | \
  perl -ne 'split; $_ =~ m/.*\/(B.*)\/.*\.wav$/ || die "Bad label $_"; print "ICSI_$1_SDM\n"' | \
   paste - $tmpdir/wav.flist > $tmpdir/wav.scp

#Keep only devset part of waves
awk '{print $2}' $tmpdir/segments | sort -u | join - $tmpdir/wav.scp | sort -o $tmpdir/wav.scp

# this file reco2file_and_channel maps recording-id (e.g. sw02001-A)
# to the file name sw02001 and the A, e.g.
# sw02001-A  sw02001 A
# In this case it's trivial, but in other corpora the information might
# be less obvious.  Later it will be needed for ctm scoring.

awk '{print $1 $2}' $tmpdir/wav.scp | \
  perl -ane '$_ =~ m:^(\S+SDM).*\/(chan.*)\.wav$: || die "bad label $_"; 
       print "$1 $2 0\n"; '\
  > $tmpdir/reco2file_and_channel || exit 1;

# we assume we adapt to the session only
awk '{print $1}' $tmpdir/segments | \
  perl -ane '$_ =~ m:^(\S+)([xfm][a-z][0-9]{3})(\S+)$: || die "bad label $_"; 
          print "$1$2$3 $1\n";'  \
    > $tmpdir/utt2spk || exit 1;

sort -k 2 $tmpdir/utt2spk | utils/utt2spk_to_spk2utt.pl > $tmpdir/spk2utt || exit 1;

# but we want to properly score the overlapped segments, hence we generate the extra
# utt2spk_stm file containing speakers ids used to generate the stms for mdm/sdm case
awk '{print $1}' $tmpdir/segments | \
  perl -ane '$_ =~ m:^(\S+)([xfm][a-z][0-9]{3})(\S+)$: || die "bad label $_"; 
          print "$1$2$3 $1$2\n";'  \
    > $tmpdir/utt2spk_stm || exit 1;


# We assume each conversation side is a separate speaker. 

# Copy stuff into its final locations [this has been moved from the format_data
# script]
mkdir -p $dir
for f in spk2utt utt2spk utt2spk_stm wav.scp text segments reco2file_and_channel; do
  cp $tmpdir/$f $dir/$f || exit 1;
done

utils/convert2stm.pl $dir utt2spk_stm > $dir/stm
cp local/english.glm $dir/glm

echo ICSI $SET set data preparation succeeded.

