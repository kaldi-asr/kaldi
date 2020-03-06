#!/usr/bin/env bash

# Copyright 2014  University of Edinburgh (Author: Pawel Swietojanski)
#           2016  Johns Hopkins University (Author: Daniel Povey)
# AMI Corpus training data preparation
# Apache 2.0

# Note: this is called by ../run.sh.

. ./path.sh

#check existing directories
if [ $# != 3 ]; then
  echo "Usage: $0 /path/to/AMI-MDM mic-name set-name"
  echo "e.g: $0 /foo/bar/AMI mdm8 dev"
  exit 1;
fi

AMI_DIR=$1
mic=$2
SET=$3

SEGS=data/local/annotations/$SET.txt
tmpdir=data/local/$mic/$SET
dir=data/$mic/${SET}_orig

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

# find selected mdm wav audio files only
find $AMI_DIR -iname "*${mic}.wav" | sort > $tmpdir/wav.flist
n=`cat $tmpdir/wav.flist | wc -l`
if [ $n -ne 169 ]; then
  echo "Warning. Expected to find 169 files but found $n."
fi

# (1a) Transcriptions preparation
awk '{meeting=$1; channel="MDM"; speaker=$3; stime=$4; etime=$5;
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
 perl -ne 'split; $_ =~ m/(.*)\_.*/; print "AMI_$1_MDM\n"' | \
  paste - $tmpdir/wav.flist > $tmpdir/wav1.scp

#Keep only devset part of waves
awk '{print $2}' $tmpdir/segments | sort -u | join - $tmpdir/wav1.scp >  $tmpdir/wav2.scp

#replace path with an appropriate sox command that select single channel only
awk '{print $1" sox -c 1 -t wavpcm -e signed-integer "$2" -t wavpcm - |"}' $tmpdir/wav2.scp > $tmpdir/wav.scp

#prep reco2file_and_channel
cat $tmpdir/wav.scp | \
  perl -ane '$_ =~ m:^(\S+MDM)\s+.*\/([IETB].*)\.wav.*$: || die "bad label $_";
       print "$1 $2 A\n"; ' > $tmpdir/reco2file_and_channel || exit 1;

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
          print "$1$2$3 $1$2\n";' > $tmpdir/utt2spk_stm || exit 1;

#check and correct case when segment timings for a given speaker overlap themself
#(important for simulatenous asclite scoring to proceed).
#There is actually only one such case for devset and automatic segmentetions
join $tmpdir/utt2spk_stm $tmpdir/segments | \
  awk '{ utt=$1; spk=$2; wav=$3; t_beg=$4; t_end=$5;
         if(spk_prev == spk && t_end_prev > t_beg) {
           print "s/^"utt, wav, t_beg, t_end"$/"utt, wav, t_end_prev, t_end"/;";
         }
         spk_prev=spk; t_end_prev=t_end;
       }' > $tmpdir/segments_to_fix

if [ -s $tmpdir/segments_to_fix ]; then
  echo "$0. Applying following fixes to segments"
  cat $tmpdir/segments_to_fix
  perl -i -pf $tmpdir/segments_to_fix $tmpdir/segments
fi

# Copy stuff into its final locations [this has been moved from the format_data
# script]
mkdir -p $dir
for f in spk2utt utt2spk utt2spk_stm wav.scp text segments reco2file_and_channel; do
  cp $tmpdir/$f $dir/$f || exit 1;
done

cp local/english.glm $dir/glm
#note, although utt2spk contains mappings to the whole meetings for simulatenous scoring
#we need to know which speakers overlap at meeting level, hence we generate an extra utt2spk_stm file
local/convert2stm.pl $dir utt2spk_stm > $dir/stm

utils/validate_data_dir.sh --no-feats $dir

echo AMI $SET set data preparation succeeded.

