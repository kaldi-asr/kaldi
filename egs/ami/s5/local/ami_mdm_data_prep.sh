#!/bin/bash

# Copyright 2014, University of Edinburgh (Author: Pawel Swietojanski)
# AMI Corpus dev/eval data preparation 

# To be run from one directory above this script.

. path.sh

#check existing directories
if [ $# != 2 ]; then
  echo "Usage: ami_data_prep.sh </path/to/AMI-MDM> <mic>"
  exit 1; 
fi 

AMI_DIR=$1
mic=$2

SEGS=data/local/annotations/train.txt
dir=data/local/$mic/train
odir=data/$mic/train
mkdir -p $dir

# Audio data directory check
if [ ! -d $AMI_DIR ]; then
  echo "Error: run.sh requires a directory argument"
  exit 1; 
fi  

# And transcripts check
if [ ! -f $SEGS ]; then
  echo "Error: File $SEGS no found (run ami_text_prep.sh)."
  exit 1;
fi

# find MDM mics
find $AMI_DIR -iname "*${mic}.wav" | sort > $dir/wav.flist

n=`cat $dir/wav.flist | wc -l`
echo "In total, $n headset files were found."
[ $n -ne 169 ] && \
  echo Warning: expected 169 data data files, found $n

# (1a) Transcriptions preparation
# here we start with rt09 transcriptions, hence not much to do

awk '{meeting=$1; channel="MDM"; speaker=$3; stime=$4; etime=$5;
 printf("AMI_%s_%s_%s_%07.0f_%07.0f", meeting, channel, speaker, int(100*stime+0.5), int(100*etime+0.5));
 for(i=6;i<=NF;i++) printf(" %s", $i); printf "\n"}' $SEGS | sort | uniq > $dir/text

# (1c) Make segment files from transcript
#segments file format is: utt-id side-id start-time end-time, e.g.:
#AMI_ES2011a_H00_FEE041_0003415_0003484
awk '{ 
       segment=$1;
       split(segment,S,"[_]");
       audioname=S[1]"_"S[2]"_"S[3]; startf=S[5]; endf=S[6];
       print segment " " audioname " " startf/100 " " endf/100 " "
}' < $dir/text > $dir/segments

#EN2001a.Array1-01.wav
#sed -e 's?.*/??' -e 's?.sph??' $dir/wav.flist | paste - $dir/wav.flist \
#  > $dir/wav.scp

sed -e 's?.*/??' -e 's?.wav??' $dir/wav.flist | \
 perl -ne 'split; $_ =~ m/(.*)\_.*/; print "AMI_$1_MDM\n"' | \
  paste - $dir/wav.flist > $dir/wav1.scp

#Keep only training part of waves
awk '{print $2}' $dir/segments | sort -u | join - $dir/wav1.scp | sort -o $dir/wav2.scp
#Two distant recordings are missing, agree segments with wav.scp
awk '{print $1}' $dir/wav2.scp | join -2 2 - $dir/segments | \
    awk '{print $2" "$1" "$3" "$4" "$5}' > $dir/s; mv $dir/s $dir/segments
#...and text with segments
awk '{print $1}' $dir/segments | join - $dir/text > $dir/t; mv $dir/t $dir/text

#replace path with an appropriate sox command that select single channel only
awk '{print $1" sox -c 1 -t wavpcm -s "$2" -t wavpcm - |"}' $dir/wav2.scp > $dir/wav.scp

#prep reco2file_and_channel
cat $dir/wav.scp | \
  perl -ane '$_ =~ m:^(\S+MDM).*\/([IETB].*)\.wav.*$: || die "bad label $_"; 
       print "$1 $2 A\n"; ' > $dir/reco2file_and_channel || exit 1;

# we assume we adapt to the session only
awk '{print $1}' $dir/segments | \
  perl -ane '$_ =~ m:^(\S+)([FM][A-Z]{0,2}[0-9]{3}[A-Z]*)(\S+)$: || die "bad label $_"; 
          print "$1$2$3 $1\n";'  \
    > $dir/utt2spk || exit 1;

sort -k 2 $dir/utt2spk | utils/utt2spk_to_spk2utt.pl > $dir/spk2utt || exit 1;

# Copy stuff into its final locations
mkdir -p $odir
for f in spk2utt utt2spk wav.scp text segments reco2file_and_channel; do
  cp $dir/$f $odir/$f | exit 1;
done

utils/validate_data_dir.sh --no-feats $odir

echo AMI MDM data preparation succeeded.

