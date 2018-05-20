#!/bin/bash

# Copyright 2014 University of Edinburgh (Author: Pawel Swietojanski)
#           2016 Johns Hopkins University (Author: Daniel Povey)
#           2017 Milos Cernak
#           2018 Emotech LTD (Author: Pawel Swietojanski)
# ICSI Corpus training data preparation
# Apache 2.0

# Note: this is called by ../run.sh.

# To be run from one directory above this script.

. ./path.sh

#check existing directories
if [ $# -ne 3 ] || [ "$2" != "ihm" ]; then
  echo "Usage: $0 /path/to/ICSI ihm set"
  echo "e.g. $0 /foo/bar/ICSI ihm set"
  echo "note: the 2nd 'ihm' argument is for compatibility with other scripts."
  exit 1;
fi

ICSI_DIR=$1
mic=$2
SET=$3

SEGS=data/local/annotations/$SET.txt
dir=data/local/$mic/$SET
odir=data/ihm/${SET}_orig
mkdir -p $dir

# Audio data directory check
if [ ! -d $ICSI_DIR ]; then
  echo "Error: $ICSI_DIR directory does not exists."
  exit 1;
fi

# And transcripts check
if [ ! -f $SEGS ]; then
  echo "Error: File $SEGS no found (run local/icsi_text_prep.sh)."
  exit 1;
fi

# (1a) Transcriptions preparation
# here we start with normalised transcriptions, the utt ids follow the convention
# ICSI_MEETING_CHAN_SPK_STIME_ETIME
# ICSI_Buw001_chan1_fe016_0003415_0003484
# we use uniq as some (rare) entries are doubled in transcripts

cat $SEGS | \
  awk '{meeting=$1; channel=$2; dchannel=$3; speaker=$4; stime=$5; etime=$6;
          if (etime > stime) {
            chan=channel;
            if (channel == "chanX") {
              split(dchannel, c, ",");
              chan = c[1];
            }
            printf("ICSI_%s_%s_%s_%07.0f_%07.0f", meeting, chan, speaker, int(100*stime+0.5), int(100*etime+0.5));
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

# Pawel: for ihm we generate wav.scp based on segments file, as we do backing off to
# distant channels for some speakers who did not have the corresponding headset.
# Also, we back off to physical directory/file names as in the ICSI data
# fetched from LDC some meeting directories starts with lowercase 'b',
# similarly, channel files names could be lowercased, i.e., chanb instead of chanB.
# No idea if this is only specific to LDC distribution, but handling it explicitly anyway.

find $ICSI_DIR/ -name "*.sph" | sort > $dir/sph.flist
awk -F'/' '{
      chan_orig=substr($NF,1,5);
      chan_norm=substr($NF,1,4)toupper(substr($NF,5,1));
      meetid_orig=substr($(NF-1),1,6);
      meetid_norm="B"substr($(NF-1),2,6);
      print "ICSI_"meetid_norm"_"chan_norm" "meetid_orig" "chan_orig;
   }' $dir/sph.flist | sort -k1 | uniq > $dir/rec2meeting_and_channel

cut -f2 -d" " $dir/segments | sort | uniq > $dir/recids0
join $dir/recids0 $dir/rec2meeting_and_channel | sort -k1 > $dir/recids

awk -v icsidir=$ICSI_DIR '{
       recid=$1;
       meetid=$2;
       chanid=$3;
       wavpath=icsidir"/"meetid"/"chanid".sph";
       print recid " " wavpath
   }' < $dir/recids > $dir/sph.scp

fsph=`head -n1 $dir/sph.scp | cut -f2 -d" "`
[ ! -f $fsph ] \
  && echo "File $f does not exist in expectetd location, make sure $ICSI_DIR is properly set" \
  && exit 1;

#add piping using sph2pipe
awk -v sph2pipe=sph2pipe '{
  printf("%s %s -f wav -p -c 1 %s |\n", $1, sph2pipe, $2); 
}' < $dir/sph.scp | sort -k1 | uniq > $dir/wav.scp || exit 1;

# (1d) reco2file_and_channel
cat $dir/wav.scp \
 | perl -ane '$_ =~ m:^ICSI_(\S+)_(\S+)\s+.*\/.*$: || die "ihm data prep: reco2file_and_channel bad label $_";
              print "ICSI_$1_$2 $1_$2 A\n"; ' > $dir/reco2file_and_channel || exit 1;

# icsi spk flags are "m", "f", "u", or "x" for male, female, unknonwn and computer generated
awk '{print $1}' $dir/segments | \
  perl -ane '$_ =~ m:^(\S+)([fmux][ne][0-9]{3})(\S+)$: || die "ihm data prep: utt2spk bad label $_";
          print "$1$2$3 $1$2\n";' > $dir/utt2spk || exit 1;

utils/utt2spk_to_spk2utt.pl <$dir/utt2spk >$dir/spk2utt || exit 1;

# Copy stuff into its final location
mkdir -p $odir
for f in spk2utt utt2spk wav.scp text segments reco2file_and_channel; do
  cp $dir/$f $odir/$f || exit 1;
done

#Produce STMs for sclite scoring
local/convert2stm.pl $dir | sort +0 -1 +1 -2 +3nb -4 > $odir/stm
cp local/english.glm $odir/glm

utils/validate_data_dir.sh --no-feats $odir || exit 1;

echo "ICSI IHM for $SET set data preparation succeeded."
