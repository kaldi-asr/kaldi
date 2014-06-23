#!/bin/bash

# Copyright 2014, University of Edinburgh (Author: Pawel Swietojanski)
# AMI Corpus dev/eval data preparation 

. path.sh

#check existing directories
if [ $# != 4 ]; then
  echo "Usage: ami_sdm_scoring_data_prep_edin.sh /path/to/AMI rt09-seg-file set-name mic"
  exit 1; 
fi 

AMI_DIR=$1
SEGS=$2 #assuming here all normalisation stuff was done
SET=$3
mic=$4

tmpdir=data/local/$mic/$SET
dir=data/$mic/$SET

mkdir -p $tmpdir

# Audio data directory check
if [ ! -d $AMI_DIR ]; then
  echo "Error: run.sh requires a directory argument"
  exit 1; 
fi  

# find headset wav audio files only, here we again get all
# the files in the corpora and filter only specific sessions
# while building segments

find $AMI_DIR -iname '*bmf[248].wav' | sort > $tmpdir/wav.flist
n=`cat $tmpdir/wav.flist | wc -l`
echo "In total, $n files were found."

# (1a) Transcriptions preparation
# here we start with rt09 transcriptions, hence not much to do

awk '{meeting=$1; channel="MDM"; speaker=$3; stime=$4; etime=$5;
 printf("AMI_%s_%s_%s_%07.0f_%07.0f", meeting, channel, speaker, int(100*stime+0.5), int(100*etime+0.5));
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

#EN2001a.Array1-01.wav
#sed -e 's?.*/??' -e 's?.sph??' $dir/wav.flist | paste - $dir/wav.flist \
#  > $dir/wav.scp

sed -e 's?.*/??' -e 's?.wav??' $tmpdir/wav.flist | \
 perl -ne 'split; $_ =~ m/(.*)_bmf([0-9])/; print "AMI_$1_MDM\n"' | \
  paste - $tmpdir/wav.flist > $tmpdir/wav.scp

#Keep only devset part of waves
awk '{print $2}' $tmpdir/segments | sort -u | join - $tmpdir/wav.scp | sort -o $tmpdir/wav.scp

# this file reco2file_and_channel maps recording-id (e.g. sw02001-A)
# to the file name sw02001 and the A, e.g.
# sw02001-A  sw02001 A
# In this case it's trivial, but in other corpora the information might
# be less obvious.  Later it will be needed for ctm scoring.

awk '{print $1 $2}' $tmpdir/wav.scp | \
  perl -ane '$_ =~ m:^(\S+MDM).*\/([IETB].*)\.wav$: || die "bad label $_"; 
       print "$1 $2 0\n"; '\
  > $tmpdir/reco2file_and_channel || exit 1;

# we assume we adapt to the session only
awk '{print $1}' $tmpdir/segments | \
  perl -ane '$_ =~ m:^(\S+)([FM][A-Z]{0,2}[0-9]{3}[A-Z]*)(\S+)$: || die "bad label $_"; 
          print "$1$2$3 $1\n";'  \
    > $tmpdir/utt2spk || exit 1;

sort -k 2 $tmpdir/utt2spk | utils/utt2spk_to_spk2utt.pl > $tmpdir/spk2utt || exit 1;

# but we want to properly score the overlapped segments, hence we generate the extra
# utt2spk_stm file containing speakers ids used to generate the stms for mdm/sdm case
awk '{print $1}' $tmpdir/segments | \
  perl -ane '$_ =~ m:^(\S+)([FM][A-Z]{0,2}[0-9]{3}[A-Z]*)(\S+)$: || die "bad label $_"; 
          print "$1$2$3 $1$2\n";'  \
    > $tmpdir/utt2spk_stm || exit 1;

# Copy stuff into its final locations [this has been moved from the format_data
# script]
mkdir -p $dir
for f in spk2utt utt2spk utt2spk_stm wav.scp text segments reco2file_and_channel; do
  cp $tmpdir/$f $dir/$f || exit 1;
done

cp local/english.glm $dir/glm
utils/convert2stm.pl $dir utt2spk_stm > $dir/stm

echo AMI $SET set data preparation succeeded.

