#!/bin/bash

# Copyright 2014  University of Edinburgh (Author: Pawel Swietojanski)
#           2016  Johns Hopkins University (Author: Daniel Povey)
#           2018  Emotech LTD (Author: Pawel Swietojanski)
# ICSI Corpus training data preparation
# Apache 2.0

. path.sh

#check existing directories
if [ $# != 3 ]; then
  echo "Usage: icsi_sdm_data_prep.sh /path/to/ICSI set micid"
  exit 1; 
fi 

CORPUS_DIR=$1
SEGS=$2 #assuming here all normalisation stuff was done
MICNUM=$3
MICID="m$MICNUM"
channels=data/local/channels.bmf

dir=data/local/sdm/$MICID/train
mkdir -p $dir

if [ ! -f $channels ]; then
  echo 'Meeting to channel mapping file missing'
  exit 1;
fi

# Audio data directory check
if [ ! -d $CORPUS_DIR ]; then
  echo "Error: run.sh requires a directory argument"
  exit 1; 
fi  

cut -d" " -f1,$(($MICNUM+1)) $channels | awk '{print $1".*"$2}' > $dir/channels_regex

# list all wav file you can
find $CORPUS_DIR -iname "*.sph" | sort > $dir/wav.flist.all
# and keep only these we want
grep -f $dir/channels_regex $dir/wav.flist.all > $dir/wav.flist

n=`cat $dir/wav.flist | wc -l`
echo "In total, $n files were found."

# (1a) Transcriptions preparation
# here we start with already normalised transcripts, just make the ids

awk '{meeting=$1; channel="SDM"; speaker=$3; stime=$4; etime=$5;
 printf("ICSI_%s_%s_%s_%07.0f_%07.0f", meeting, channel, speaker, int(100*stime+0.5), int(100*etime+0.5));
 for(i=6;i<=NF;i++) printf(" %s", $i); printf "\n"}' $SEGS | sort  > $dir/text

# (1c) Make segment files from transcript
#segments file format is: utt-id side-id start-time end-time, e.g.:
#AMI_ES2011a_H00_FEE041_0003415_0003484
awk '{ 
       segment=$1;
       split(segment,S,"[_]");
       audioname=S[1]"_"S[2]"_"S[3]; startf=S[5]; endf=S[6];
       print segment " " audioname " " startf/100 " " endf/100 " " 0
}' < $dir/text > $dir/segments

#EN2001a.Array1-01.wav
#sed -e 's?.*/??' -e 's?.sph??' $dir/wav.flist | paste - $dir/wav.flist \
#  > $dir/wav.scp

cat $dir/wav.flist | \
 perl -ne 'split; $_ =~ m/.*\/(B.*)\/.*\.wav/; print "ICSI_$1_SDM\n"' | \
  paste - $dir/wav.flist > $dir/wav.scp

awk '{print $1 $2}' $dir/wav.scp | \
  perl -ane '$_ =~ m:^(\S+SDM).*\/(chan.*)\.wav$: || die "bad label $_"; 
       print "$1 $2 0\n"; '\
  > $dir/reco2file_and_channel || exit 1;

# we assume we adapt to the session only
awk '{print $1}' $dir/segments | \
  perl -ane '$_ =~ m:^(\S+)([fmx][a-z][0-9]{3})(\S+)$: || die "bad label $_"; 
          print "$1$2$3 $1\n";'  \
    > $dir/utt2spk || exit 1;

sort -k 2 $dir/utt2spk | utils/utt2spk_to_spk2utt.pl > $dir/spk2utt || exit 1;

# Copy stuff into its final locations [this has been moved from the format_data
# script]
mkdir -p data/sdm/$MICID/train
for f in spk2utt utt2spk wav.scp text segments reco2file_and_channel; do
  cp $dir/$f data/sdm/$MICID/train/$f || exit 1;
done

echo ICSI data preparation succeeded.

