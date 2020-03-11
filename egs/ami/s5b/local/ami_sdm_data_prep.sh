#!/usr/bin/env bash


# Copyright 2014  University of Edinburgh (Author: Pawel Swietojanski)
#           2016  Johns Hopkins University (Author: Daniel Povey)
# AMI Corpus training data preparation
# Apache 2.0

# Note: this is called by ../run.sh.

. ./path.sh

#check existing directories
if [ $# != 2 ]; then
  echo "Usage: ami_sdm_data_prep.sh <path/to/AMI> <mic-id>"
  echo "e.g.: $0 /foo/bar/ami sdm1"
  exit 1;
fi

AMI_DIR=$1
MICNUM=$(echo $2 | sed s/[a-z]//g)
DSET="sdm$MICNUM"

if [ "$DSET" != "$2" ]; then
  echo "$0: bad 2nd argument: $*"
  exit 1
fi

SEGS=data/local/annotations/train.txt
dir=data/local/$DSET/train
odir=data/$DSET/train_orig
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

# as the sdm we treat first mic from the array
find $AMI_DIR -iname "*.Array1-0$MICNUM.wav" | sort > $dir/wav.flist

n=`cat $dir/wav.flist | wc -l`

echo "In total, $n files were found."
[ $n -ne 169 ] && \
  echo Warning: expected 169 data data files, found $n

# (1a) Transcriptions preparation
# here we start with already normalised transcripts, just make the ids
# Note, we set here SDM rather than, for example, SDM1 as we want to easily use
# the same alignments across different mics

awk '{meeting=$1; channel="SDM"; speaker=$3; stime=$4; etime=$5;
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

sed -e 's?.*/??' -e 's?.wav??' $dir/wav.flist | \
 perl -ne 'split; $_ =~ m/(.*)\..*/; print "AMI_$1_SDM\n"' | \
  paste - $dir/wav.flist > $dir/wav1.scp

#Keep only training part of waves
awk '{print $2}' $dir/segments | sort -u | join - $dir/wav1.scp | sort -o $dir/wav2.scp
#Two distant recordings are missing, agree segments with wav.scp
awk '{print $1}' $dir/wav2.scp | join -2 2 - $dir/segments | \
    awk '{print $2" "$1" "$3" "$4" "$5}' > $dir/s; mv $dir/s $dir/segments
#...and text with segments
awk '{print $1}' $dir/segments | join - $dir/text > $dir/t; mv $dir/t $dir/text

#replace path with an appropriate sox command that select a single channel only
awk '{print $1" sox -c 1 -t wavpcm -e signed-integer "$2" -t wavpcm - |"}' $dir/wav2.scp > $dir/wav.scp

# this file reco2file_and_channel maps recording-id
cat $dir/wav.scp | \
  perl -ane '$_ =~ m:^(\S+SDM)\s+.*\/([IETB].*)\.wav.*$: || die "bad label $_";
       print "$1 $2 A\n"; ' > $dir/reco2file_and_channel || exit 1;

# In this data-prep phase we adapt to the session only [later on we may split
# into shorter pieces].
# We use the first two underscore-separated fields of the utterance-id
# as the speaker-id, e.g. 'AMI_EN2001a_SDM_FEO065_0090130_0090775' becomes 'AMI_EN2001a'.
awk '{print $1}' $dir/segments | \
  perl -ane 'chop; @A = split("_", $_); $spkid = join("_", @A[0,1]); print "$_ $spkid\n";'  \
  >$dir/utt2spk || exit 1;

utils/utt2spk_to_spk2utt.pl <$dir/utt2spk >$dir/spk2utt || exit 1;

# Copy stuff into its final locations
mkdir -p $odir
for f in spk2utt utt2spk wav.scp text segments reco2file_and_channel; do
  cp $dir/$f $odir/$f || exit 1;
done

utils/validate_data_dir.sh --no-feats $odir

echo AMI $DSET data preparation succeeded.

