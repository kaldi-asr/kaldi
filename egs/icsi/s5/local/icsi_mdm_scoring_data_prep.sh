#!/bin/bash

# Copyright 2014 University of Edinburgh (Author: Pawel Swietojanski)
#           2016 Johns Hopkins University (Author: Daniel Povey)
#           2018 Emotech LTD (Author: Pawel Swietojanski)
# ICSI Corpus training data preparation
# Apache 2.0

# Note: this is called by ../run.sh.

# To be run from one directory above this script.

. ./path.sh

#check existing directories
if [ $# -ne 3 ]; then
  echo "Usage: $0 /path/to/ICSI mdmX set"
  echo "e.g. $0 /foo/bar/ICSI mdm4 eval"
  exit 1;
fi

ICSI_DIR=$1
mic=$2
nmics=$(echo $mic | sed 's/[a-z]//g')
SET=$3

SEGS=data/local/annotations/$SET.txt
dir=data/local/$mic/$SET
odir=data/$mic/${SET}_orig
mkdir -p $dir

# Audio data directory check
if [ ! -d $ICSI_DIR ]; then
  echo "Error: $ICSI_DIR directory does not exists."
  exit 1;
fi

# And transcripts check
if [ ! -f $SEGS ]; then
  echo "Error: File $SEGS no found (run icsi_text_prep.sh)."
  exit 1;
fi

# (1a) Transcriptions preparation
# here we start with normalised transcriptions, the utt ids follow the convention
# ICSI_MEETING_CHAN_SPK_STIME_ETIME
# ICSI_Buw001_mdm4_fe016_0003415_0003484
# we use uniq as some (rare) entries are doubled in transcripts

cat $SEGS | \
  awk -v micdir=$mic \
      '{meeting=$1; channel=$2; dchannel=$3; speaker=$4; stime=$5; etime=$6;
          if (etime > stime) {
            printf("ICSI_%s_%s_%s_%07.0f_%07.0f", meeting, micdir, speaker, int(100*stime+0.5), int(100*etime+0.5));
            for(i=7;i<=NF;i++) printf(" %s", $i); printf "\n";
          }
       }' | sort -k1 | uniq > $dir/text

# (1b) Make segment files from transcript
awk '{
       segment=$1;
       split(segment,S,"[_]");
       audioname=S[1]"_"S[2]"_"S[3]; startf=S[5]; endf=S[6];
       print segment " " audioname " " startf*10/1000 " " endf*10/1000 " "
}' < $dir/text > $dir/segments

# (1c) prepare wav.scp
#find $ICSI_DIR/ -name "*$mic.wav" | sort > $dir/wav.flist

awk -v icsidir=$ICSI_DIR -v mic=$mic \
  '{ uttid=$1; recid=$2; start=$3; stop=$4;
     split(recid, R, "[_]");
     meetid=R[2]
     wavpath=icsidir"/"meetid"/"meetid"_"mic".wav";
     print recid " " wavpath
   }' < $dir/segments | sort -k1 | uniq > $dir/wav.scp

nws=`cat $dir/wav.scp | wc -l`

#nwf=`cat $dir/wav.flist | wc -l`
#[ $nws -ne $nwf ] && echo "Expected num of wavs@ $nws, got: $nwf. Check beamforming stage" && exit 1; 

fsph=`head -n1 $dir/wav.scp | cut -f2 -d" "`
[ ! -f $fsph ] \
  && echo "File $fsph does not exist in expectetd location, make sure $ICSI_DIR is properly set" \
  && exit 1;

# (1d) reco2file_and_channel
cat $dir/wav.scp \
 | perl -ane '$_ =~ m:^ICSI_(\S+)_(\S+)\s+.*\.wav$: || die "mdm data prep: reco2file_and_channel bad label $_";
              print "ICSI_$1_$2 $1_$2 A\n"; ' > $dir/reco2file_and_channel || exit 1;

# icsi spk flags are "m", "f", "u", or "x" for male, female, unknonwn and computer generated
# for distant case, we do not include speaker label
awk '{print $1}' $dir/segments | \
  perl -ane '$_ =~ m:^(\S+)([fmux][ne][0-9]{3})(\S+)$: || die "mdm data prep: utt2spk bad label $_";
          print "$1$2$3 $1\n";' > $dir/utt2spk || exit 1;

utils/utt2spk_to_spk2utt.pl <$dir/utt2spk >$dir/spk2utt || exit 1;

# but we want to properly score the overlapped segments, hence we generate the extra
# utt2spk_stm file containing speakers ids used to generate the stms for mdm/sdm case
awk '{print $1}' $dir/segments | \
  perl -ane '$_ =~ m:^(\S+)([fmux][ne][0-9]{3})(\S+)$: || die "mdm data prep: utt2spk_stm bad label $_";
          print "$1$2$3 $1$2\n";' > $dir/utt2spk_stm || exit 1;

local/convert2stm.pl $dir utt2spk_stm | sort +0 -1 +1 -2 +3nb -4 > $dir/stm
cp local/english.glm $dir/glm

# Copy stuff into its final location
mkdir -p $odir
for f in spk2utt utt2spk wav.scp text segments reco2file_and_channel stm glm; do
  cp $dir/$f $odir/$f || exit 1;
done

utils/validate_data_dir.sh --no-feats $odir || exit 1;

echo "ICSI $mic for $SET data preparation succeeded."
