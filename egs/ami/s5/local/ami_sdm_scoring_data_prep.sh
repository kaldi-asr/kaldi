#!/bin/bash

# Copyright 2014, University of Edinburgh (Author: Pawel Swietojanski)
# AMI Corpus dev/eval data preparation

. path.sh

#check existing directories
if [ $# != 3 ]; then
  echo "Usage: ami_sdm_scoring_data_prep.sh <path/to/AMI> <mic-id> <set-name>"
  exit 1;
fi

AMI_DIR=$1
MICNUM=$2
SET=$3
DSET="sdm$MICNUM"

SEGS=data/local/annotations/$SET.txt
tmpdir=data/local/$DSET/$SET
dir=data/$DSET/$SET

mkdir -p $tmpdir

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

find $AMI_DIR -iname "*.Array1-0$MICNUM.wav" | sort > $tmpdir/wav.flist

n=`cat $tmpdir/wav.flist | wc -l`
echo "In total, $n files were found."

# (1a) Transcriptions preparation
# here we start with normalised transcripts

awk '{meeting=$1; channel="SDM"; speaker=$3; stime=$4; etime=$5;
 printf("AMI_%s_%s_%s_%07.0f_%07.0f", meeting, channel, speaker, int(100*stime+0.5), int(100*etime+0.5));
 for(i=6;i<=NF;i++) printf(" %s", $i); printf "\n"}' $SEGS | sort | uniq > $tmpdir/text

# (1c) Make segment files from transcript
#segments file format is: utt-id side-id start-time end-time, e.g.:
#AMI_ES2011a_H00_FEE041_0003415_0003484
awk '{
       segment=$1;
       split(segment,S,"[_]");
       audioname=S[1]"_"S[2]"_"S[3]; startf=S[5]; endf=S[6];
       print segment " " audioname " " startf/100 " " endf/100 " "
}' < $tmpdir/text > $tmpdir/segments

#EN2001a.Array1-01.wav
#sed -e 's?.*/??' -e 's?.sph??' $dir/wav.flist | paste - $dir/wav.flist \
#  > $dir/wav.scp

sed -e 's?.*/??' -e 's?.wav??' $tmpdir/wav.flist | \
 perl -ne 'split; $_ =~ m/(.*)\..*/; print "AMI_$1_SDM\n"' | \
  paste - $tmpdir/wav.flist > $tmpdir/wav1.scp

#Keep only devset part of waves
awk '{print $2}' $tmpdir/segments | sort -u | join - $tmpdir/wav1.scp > $tmpdir/wav2.scp

#replace path with an appropriate sox command that select single channel only
awk '{print $1" sox -c 1 -t wavpcm -s "$2" -t wavpcm - |"}' $tmpdir/wav2.scp > $tmpdir/wav.scp

#prep reco2file_and_channel
cat $tmpdir/wav.scp | \
  perl -ane '$_ =~ m:^(\S+SDM).*\/([IETB].*)\.wav.*$: || die "bad label $_";
       print "$1 $2 A\n"; '\
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

#check and correct the case when segment timings for given speaker overlap themself
#(important for simulatenous asclite scoring to proceed).
#There is actually only one such case for devset and automatic segmentetions
join $tmpdir/utt2spk_stm $tmpdir/segments | \
  awk '{ utt=$1; spk=$2; wav=$3; t_beg=$4; t_end=$5;
         if(spk_prev == spk && t_end_prev > t_beg) {
           print utt, wav, t_beg, t_end">"utt, wav, t_end_prev, t_end;
         }
         spk_prev=spk; t_end_prev=t_end;
       }' > $tmpdir/segments_to_fix

if [ `cat $tmpdir/segments_to_fix | wc -l` -gt 0 ]; then
  echo "$0. Applying following fixes to segments"
  cat $tmpdir/segments_to_fix
  while read line; do
     p1=`echo $line | awk -F'>' '{print $1}'`
     p2=`echo $line | awk -F'>' '{print $2}'`
     sed -ir "s:$p1:$p2:" $tmpdir/segments
  done < $tmpdir/segments_to_fix
fi

# Copy stuff into its final locations [this has been moved from the format_data
# script]
mkdir -p $dir
for f in spk2utt utt2spk utt2spk_stm wav.scp text segments reco2file_and_channel; do
  cp $tmpdir/$f $dir/$f || exit 1;
done

local/convert2stm.pl $dir utt2spk_stm > $dir/stm
cp local/english.glm $dir/glm

utils/validate_data_dir.sh --no-feats $dir

echo AMI $DSET scenario and $SET set data preparation succeeded.

