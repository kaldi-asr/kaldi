#!/bin/bash

# Copyright 2014, University of Edinburgh (Author: Pawel Swietojanski)
# AMI Corpus training data preparation 
# Apache 2.0

# To be run from one directory above this script.

. path.sh

#check existing directories
if [ $# != 2 ]; then
  echo "Usage: ami_data_prep_edin.sh /path/to/AMI"
  exit 1; 
fi 

AMI_DIR=$1
RT09_SEGS=$2 #assuming here all normalisation stuff was done

dir=data/local/ihm/train
mkdir -p $dir

# Audio data directory check
if [ ! -d $AMI_DIR ]; then
  echo "Error: run.sh requires a directory argument"
  exit 1; 
fi  

# find headset wav audio files only
find $AMI_DIR -iname '*.Headset-*.wav' | sort > $dir/wav.flist

n=`cat $dir/wav.flist | wc -l`

echo "In total, $n headset files were found."
#[ $n -ne 2435 ] && \
#  echo Warning: expected 2435 data data files, found $n

# (1a) Transcriptions preparation
# here we start with rt09 transcriptions, hence not much to do

cut -d" " -f1,4- $RT09_SEGS | sort > $dir/text

# **NOTE: swbd1_map_words.pl has been modified to make the pattern matches 
# case insensitive
#local/swbd1_map_words.pl -f 2- $dir/transcripts2.txt > $dir/text  # final transcripts

# (1c) Make segment files from transcript
#segments file format is: utt-id side-id start-time end-time, e.g.:
#AMI_ES2011a_H00_FEE041_0003415_0003484
awk '{ 
       segment=$1;
       split(segment,S,"[_]");
       audioname=S[1]"_"S[2]"_"S[3]; startf=S[5]; endf=S[6];
       print segment " " audioname " " startf*10/1000 " " endf*10/1000 " " 0
}' < $dir/text > $dir/segments

#sw02001-A_000098-001156 sw02001-A 0.98 11.56
#awk '{ 
#       segment=$1;
#       split(segment,S,"[_-]");
#       side=S[2]; audioname=S[1]; startf=S[3]; endf=S[4];
#       print segment " " audioname "-" side " " startf/100 " " endf/100
#}' < $dir/text > $dir/segments

#EN2001a.Headset-0.wav
#sed -e 's?.*/??' -e 's?.sph??' $dir/wav.flist | paste - $dir/wav.flist \
#  > $dir/wav.scp

sed -e 's?.*/??' -e 's?.wav??' $dir/wav.flist | \
 perl -ne 'split; $_ =~ m/(.*)\..*\-([0-9])/; print "AMI_$1_H0$2\n"' | \
  paste - $dir/wav.flist > $dir/wav.scp

# this file reco2file_and_channel maps recording-id (e.g. sw02001-A)
# to the file name sw02001 and the A, e.g.
# sw02001-A  sw02001 A
# In this case it's trivial, but in other corpora the information might
# be less obvious.  Later it will be needed for ctm scoring.

awk '{print $1}' $dir/wav.scp \
  | perl -ane '$_ =~ m:^(\S+)(H0[0-4])$: || die "bad label $_"; 
               print "$1$2 $1$2 $2\n"; ' \
  > $dir/reco2file_and_channel || exit 1;

awk '{print $1}' $dir/segments | \
  perl -ane '$_ =~ m:^(\S+)([FM][A-Z]{0,2}[0-9]{3}[A-Z]*)(\S+)$: || die "bad label $_"; 
          print "$1$2$3 $1$2\n";'  \
    > $dir/utt2spk || exit 1;

sort -k 2 $dir/utt2spk | utils/utt2spk_to_spk2utt.pl > $dir/spk2utt || exit 1;

# We assume each conversation side is a separate speaker. This is a very 
# reasonable assumption for Switchboard. The actual speaker info file is at:
# http://www.ldc.upenn.edu/Catalog/desc/addenda/swb-multi-annot.summary

# Copy stuff into its final locations [this has been moved from the format_data
# script]
mkdir -p data/ihm/train
for f in spk2utt utt2spk wav.scp text segments reco2file_and_channel; do
  cp $dir/$f data/ihm/train/$f || exit 1;
done

echo AMI data preparation succeeded.

