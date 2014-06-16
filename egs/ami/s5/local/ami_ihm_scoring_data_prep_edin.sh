#!/bin/bash

# Switchboard-1 training data preparation customized for Edinburgh
# Author:  Arnab Ghoshal (Jan 2013)

# To be run from one directory above this script.

## The input is some directory containing the switchboard-1 release 2
## corpus (LDC97S62).  Note: we don't make many assumptions about how
## you unpacked this.  We are just doing a "find" command to locate
## the .sph files.

. path.sh

#check existing directories
if [ $# != 3 ]; then
  echo "Usage: scoring_data_prep_edin.sh /path/to/SWBD rt09-seg-file set-name"
  exit 1; 
fi 

AMI_DIR=$1
RT09_SEGS=$2 #assuming here all normalisation stuff was done
SET=$3

tmpdir=data/local/ihm/$SET
dir=data/ihm/$SET

mkdir -p $tmpdir

# Audio data directory check
if [ ! -d $AMI_DIR ]; then
  echo "Error: run.sh requires a directory argument"
  exit 1; 
fi  

# find headset wav audio files only, here we again get all
# the files in the corpora and filter only specific sessions
# while building segments

find $AMI_DIR -iname '*.Headset-*.wav' | sort > $tmpdir/wav.flist
n=`cat $tmpdir/wav.flist | wc -l`
echo "In total, $n headset files were found."

# (1a) Transcriptions preparation
# here we start with rt09 transcriptions, hence not much to do

cut -d" " -f1,4- $RT09_SEGS | sort > $tmpdir/text

# (1c) Make segment files from transcript
#segments file format is: utt-id side-id start-time end-time, e.g.:
#AMI_ES2011a_H00_FEE041_0003415_0003484
awk '{ 
       segment=$1;
       split(segment,S,"[_]");
       audioname=S[1]"_"S[2]"_"S[3]; startf=S[5]; endf=S[6];
       print segment " " audioname " " startf*10/1000 " " endf*10/1000 " " 0
}' < $tmpdir/text > $tmpdir/segments

#EN2001a.Headset-0.wav
#sed -e 's?.*/??' -e 's?.sph??' $dir/wav.flist | paste - $dir/wav.flist \
#  > $dir/wav.scp

sed -e 's?.*/??' -e 's?.wav??' $tmpdir/wav.flist | \
 perl -ne 'split; $_ =~ m/(.*)\..*\-([0-9])/; print "AMI_$1_H0$2\n"' | \
  paste - $tmpdir/wav.flist > $tmpdir/wav.scp

#Keep only devset part of waves
awk '{print $2}' $tmpdir/segments | sort -u | join - $tmpdir/wav.scp | sort -o $tmpdir/wav.scp

# this file reco2file_and_channel maps recording-id (e.g. sw02001-A)
# to the file name sw02001 and the A, e.g.
# sw02001-A  sw02001 A
# In this case it's trivial, but in other corpora the information might
# be less obvious.  Later it will be needed for ctm scoring.

awk '{print $1 $2}' $tmpdir/wav.scp | \
  perl -ane '$_ =~ m:^(\S+H0[0-4]).*\/([IETB].*)\.wav$: || die "bad label $_"; 
       print "$1 $2 0\n"; '\
  > $tmpdir/reco2file_and_channel || exit 1;

awk '{print $1}' $tmpdir/segments | \
  perl -ane '$_ =~ m:^(\S+)([FM][A-Z]{0,2}[0-9]{3}[A-Z]*)(\S+)$: || die "bad label $_"; 
          print "$1$2$3 $1$2\n";'  \
    > $tmpdir/utt2spk || exit 1;

sort -k 2 $tmpdir/utt2spk | utils/utt2spk_to_spk2utt.pl > $tmpdir/spk2utt || exit 1;

# We assume each conversation side is a separate speaker. 

# Copy stuff into its final locations [this has been moved from the format_data
# script]
mkdir -p $dir
for f in spk2utt utt2spk wav.scp text segments reco2file_and_channel; do
  cp $tmpdir/$f $dir/$f || exit 1;
done

utils/convert2stm.pl $dir > $dir/stm
cp local/english.glm $dir/glm

echo AMI $SET set data preparation succeeded.

