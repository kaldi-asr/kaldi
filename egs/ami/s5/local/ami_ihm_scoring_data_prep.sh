#!/bin/bash

# Copyright 2014, University of Edinburgh (Author: Pawel Swietojanski)
# AMI Corpus dev/eval data preparation 

. path.sh

#check existing directories
if [ $# != 2 ]; then
  echo "Usage: ami_*_scoring_data_prep_edin.sh /path/to/AMI  set-name"
  exit 1; 
fi 

AMI_DIR=$1
SET=$2
SEGS=data/local/annotations/$SET.txt

dir=data/local/ihm/$SET
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

# find headset wav audio files only, here we again get all
# the files in the corpora and filter only specific sessions
# while building segments

find $AMI_DIR -iname '*.Headset-*.wav' | sort > $dir/wav.flist
n=`cat $dir/wav.flist | wc -l`
echo "In total, $n headset files were found."
[ $n -ne 687 ] && \
  echo "Warning: expected 687 (168 mtgs x 4 mics + 3 mtgs x 5 mics) data files, found $n"

# (1a) Transcriptions preparation
# here we start with normalised transcriptions, the utt ids follow the convention
# AMI_MEETING_CHAN_SPK_STIME_ETIME
# AMI_ES2011a_H00_FEE041_0003415_0003484

awk '{meeting=$1; channel=$2; speaker=$3; stime=$4; etime=$5;
 printf("AMI_%s_%s_%s_%07.0f_%07.0f", meeting, channel, speaker, int(100*stime+0.5), int(100*etime+0.5));
 for(i=6;i<=NF;i++) printf(" %s", $i); printf "\n"}' $SEGS | sort | uniq > $dir/text

# (1c) Make segment files from transcript
#segments file format is: utt-id side-id start-time end-time, e.g.:

awk '{ 
       segment=$1;
       split(segment,S,"[_]");
       audioname=S[1]"_"S[2]"_"S[3]; startf=S[5]; endf=S[6];
       print segment " " audioname " " startf*10/1000 " " endf*10/1000 " "
}' < $dir/text > $dir/segments

#prepare wav.scp
sed -e 's?.*/??' -e 's?.wav??' $dir/wav.flist | \
 perl -ne 'split; $_ =~ m/(.*)\..*\-([0-9])/; print "AMI_$1_H0$2\n"' | \
  paste - $dir/wav.flist > $dir/wav1.scp

#Keep only  train part of waves
awk '{print $2}' $dir/segments | sort -u | join - $dir/wav1.scp >  $dir/wav2.scp

#replace path with an appropriate sox command that select single channel only
awk '{print $1" sox -c 1 -t wavpcm -s "$2" -t wavpcm - |"}' $dir/wav2.scp > $dir/wav.scp

# (1d) reco2file_and_channel
cat $dir/wav.scp \
 | perl -ane '$_ =~ m:^(\S+)(H0[0-4])\s+.*\/([IETB].*)\.wav.*$: || die "bad label $_"; 
              print "$1$2 $3 A\n"; ' > $dir/reco2file_and_channel || exit 1;

awk '{print $1}' $dir/segments | \
  perl -ane '$_ =~ m:^(\S+)([FM][A-Z]{0,2}[0-9]{3}[A-Z]*)(\S+)$: || die "segments: bad label $_"; 
          print "$1$2$3 $1$2\n";' > $dir/utt2spk || exit 1;

sort -k 2 $dir/utt2spk | utils/utt2spk_to_spk2utt.pl > $dir/spk2utt || exit 1;

#check and correct the case when segment timings for given speaker overlap themself 
#(important for simulatenous asclite scoring to proceed).
#There is actually only one such case for devset and automatic segmentetions
join $dir/utt2spk $dir/segments | \
   perl -ne '{BEGIN{$pu=""; $pt=0.0;} split;
           if ($pu eq $_[1] && $pt > $_[3]) {
             print "$_[0] $_[2] $_[3] $_[4]>$_[0] $_[2] $pt $_[4]\n"
           }
           $pu=$_[1]; $pt=$_[4]; 
         }' > $dir/segments_to_fix
if [ `cat $dir/segments_to_fix | wc -l` -gt 0 ]; then
  echo "$0. Applying following fixes to segments"
  cat $dir/segments_to_fix
  while read line; do
     p1=`echo $line | awk -F'>' '{print $1}'`
     p2=`echo $line | awk -F'>' '{print $2}'`
     sed -ir "s!$p1!$p2!" $dir/segments
  done < $dir/segments_to_fix
fi

# Copy stuff into its final locations
fdir=data/ihm/$SET
mkdir -p $fdir
for f in spk2utt utt2spk wav.scp text segments reco2file_and_channel; do
  cp $dir/$f $fdir/$f || exit 1;
done

#Produce STMs for sclite scoring
local/convert2stm.pl $dir > $fdir/stm
cp local/english.glm $fdir/glm

utils/validate_data_dir.sh --no-feats $fdir || exit 1;

echo AMI $SET set data preparation succeeded.

