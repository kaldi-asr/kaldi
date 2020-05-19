#!/usr/bin/env bash

# Copyright 2014, University of Edinburgh (Author: Pawel Swietojanski)
# AMI Corpus training data preparation 
# Apache 2.0

# To be run from one directory above this script.

. ./path.sh

#check existing directories
if [ $# != 1 ]; then
  echo "Usage: ami_ihm_data_prep.sh /path/to/AMI"
  exit 1; 
fi 

AMI_DIR=$1

SEGS=data/local/annotations/train.txt
dir=data/local/ihm/train
mkdir -p $dir

# Audio data directory check
if [ ! -d $AMI_DIR ]; then
  echo "Error: $AMI_DIR directory does not exists."
  exit 1; 
fi  

# And transcripts check
if [ ! -f $SEGS ]; then
  echo "Error: File $SEGS no found (run ami_text_prep.sh)."
  exit 1;
fi


# find headset wav audio files only
find $AMI_DIR -iname '*.Headset-*.wav' | sort > $dir/wav.flist
n=`cat $dir/wav.flist | wc -l`
echo "In total, $n headset files were found."
[ $n -ne 687 ] && \
  echo "Warning: expected 687 (168 mtgs x 4 mics + 3 mtgs x 5 mics) data files, found $n"

# (1a) Transcriptions preparation
# here we start with normalised transcriptions, the utt ids follow the convention
# AMI_MEETING_CHAN_SPK_STIME_ETIME
# AMI_ES2011a_H00_FEE041_0003415_0003484
# we use uniq as some (rare) entries are doubled in transcripts

awk '{meeting=$1; channel=$2; speaker=$3; stime=$4; etime=$5;
 printf("AMI_%s_%s_%s_%07.0f_%07.0f", meeting, channel, speaker, int(100*stime+0.5), int(100*etime+0.5));
 for(i=6;i<=NF;i++) printf(" %s", $i); printf "\n"}' $SEGS | sort | uniq > $dir/text

# (1b) Make segment files from transcript

awk '{ 
       segment=$1;
       split(segment,S,"[_]");
       audioname=S[1]"_"S[2]"_"S[3]; startf=S[5]; endf=S[6];
       print segment " " audioname " " startf*10/1000 " " endf*10/1000 " "
}' < $dir/text > $dir/segments

# (1c) Make wav.scp file.

sed -e 's?.*/??' -e 's?.wav??' $dir/wav.flist | \
 perl -ne 'split; $_ =~ m/(.*)\..*\-([0-9])/; print "AMI_$1_H0$2\n"' | \
  paste - $dir/wav.flist > $dir/wav1.scp

#Keep only  train part of waves
awk '{print $2}' $dir/segments | sort -u | join - $dir/wav1.scp >  $dir/wav2.scp

#replace path with an appropriate sox command that select single channel only
awk '{print $1" sox -c 1 -t wavpcm -e signed-integer "$2" -t wavpcm - |"}' $dir/wav2.scp > $dir/wav.scp

# (1d) reco2file_and_channel
cat $dir/wav.scp \
 | perl -ane '$_ =~ m:^(\S+)(H0[0-4])\s+.*\/([IETB].*)\.wav.*$: || die "bad label $_"; 
              print "$1$2 $3 A\n"; ' > $dir/reco2file_and_channel || exit 1;


awk '{print $1}' $dir/segments | \
  perl -ane '$_ =~ m:^(\S+)([FM][A-Z]{0,2}[0-9]{3}[A-Z]*)(\S+)$: || die "bad label $_"; 
          print "$1$2$3 $1$2\n";' > $dir/utt2spk || exit 1;

sort -k 2 $dir/utt2spk | utils/utt2spk_to_spk2utt.pl > $dir/spk2utt || exit 1;

# Copy stuff into its final location
mkdir -p data/ihm/train
for f in spk2utt utt2spk wav.scp text segments reco2file_and_channel; do
  cp $dir/$f data/ihm/train/$f || exit 1;
done

utils/validate_data_dir.sh --no-feats data/ihm/train || exit 1;

echo AMI IHM data preparation succeeded.

