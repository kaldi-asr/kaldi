#!/usr/bin/env bash
#

# To be run from one directory above this script.

## The input is some directory containing the switchboard-1 release 2
## corpus (LDC97S62).  Note: we don't make many assumptions about how
## you unpacked this.  We are just doing a "find" command to locate
## the .sph files.

# for example /mnt/matylda2/data/SWITCHBOARD_1R2

. ./path.sh

#check existing directories
if [ $# != 1 ]; then
   echo "Usage: swbd_p1_data_prep.sh /path/to/SWBD"
   exit 1; 
fi 

SWBD_DIR=$1

dir=data/local/train
mkdir -p $dir


# Audio data directory check
if [ ! -d $SWBD_DIR ]; then
  echo "Error: run.sh requires a directory argument"
  exit 1; 
fi  

# Trans directory check
if [ ! -d $dir/swb_ms98_transcriptions ]; then
   # To get the SWBD transcriptions and dict, do:
   echo " *** Downloading trascriptions and dictionary ***"   
   ( 
    cd $dir;
    wget http://www.isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz
    tar -xf switchboard_word_alignments.tar.gz
   )
else
  echo "Directory with transcriptions exists, skipping downloading"
fi


# Option A: SWBD dictionary file check
[ ! -f $dir/swb_ms98_transcriptions/sw-ms98-dict.text ] && \
     echo  "SWBD dictionary file does not exist" &&  exit 1;

# find sph audio files
find $SWBD_DIR -iname '*.sph' > $dir/sph.flist

n=`cat $dir/sph.flist | wc -l`
[ $n -ne 2435 ] && \
  echo Warning: expected 2435 data data files, found $n


# (1a) Transcriptions preparation
# make basic transcription file (add segments info)
awk '{name=substr($1,1,6);gsub("^sw","sw0",name); side=substr($1,7,1);stime=$2;etime=$3;
 printf("%s-%s_%06.0f-%06.0f", name, side, int(100*stime+0.5), int(100*etime+0.5));
 for(i=4;i<=NF;i++) printf(" %s", toupper($i)); printf "\n"}' \
  $dir/swb_ms98_transcriptions/*/*/*-trans.text  > $dir/transcripts1.txt

# test if trans. file is sorted
export LC_ALL=C;
sort -c $dir/transcripts1.txt || exit 1; # check it's sorted.

# Remove SILENCE, <B_ASIDE> and <E_ASIDE>.

# Note: we have [NOISE], [VOCALIZED-NOISE], [LAUGHTER], [SILENCE].
# removing [SILENCE], and the <B_ASIDE> and <E_ASIDE> markers that mark
# speech to somone; we will give phones to the other three (NSN, SPN, LAU). 
# There will also be a silence phone, SIL.

cat $dir/transcripts1.txt | perl -ane 's:\s\[SILENCE\](\s|$):$1:g; s/<B_ASIDE>//g; s/<E_ASIDE>//g; print; ' | \
  awk '{if(NF > 1) { print; } } ' > $dir/transcripts2.txt

local/swbd_map_words.pl -f 2- $dir/transcripts2.txt > $dir/text  # This is the final transcripts...

# (1c) Make segment files from transcript
#segments file format is: utt-id side-id start-time end-time, e.g.:
#sw02001-A_000098-001156 sw02001-A 0.98 11.56

awk '{ segment=$1; split(segment,S,"[_-]"); side=S[2]; audioname=S[1];startf=S[3];endf=S[4];
   print segment " " audioname "-" side " " startf/100 " " endf/100}' <$dir/text > $dir/segments

awk '{name = $0; gsub(".sph$","",name); gsub(".*/","",name); print(name " " $0)}' $dir/sph.flist > $dir/sph.scp

sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
[ ! -f $sph2pipe ] && echo "Could not find the sph2pipe program at $sph2pipe" && exit 1;

cat $dir/sph.scp | awk -v sph2pipe=$sph2pipe '{printf("%s-A %s -f wav -p -c 1 %s |\n", $1, sph2pipe, $2); 
    printf("%s-B %s -f wav -p -c 2 %s |\n", $1, sph2pipe, $2);}' | \
   sort > $dir/wav.scp || exit 1;
 #side A - channel 1, side B - channel 2

# this file reco2file_and_channel maps recording-id (e.g. sw02001-A)
# to the file name sw02001 and the A, e.g.
# sw02001-A  sw02001 A
# In this case it's trivial, but in other corpora the information might
# be less obvious.  Later it will be needed for ctm scoring.
cat $dir/wav.scp | awk '{print $1}' | \
  perl -ane '$_ =~ m:^(\S+)-([AB])$: || die "bad label $_"; print "$1-$2 $1 $2\n"; ' \
  > $dir/reco2file_and_channel || exit 1;

cat $dir/segments | awk '{spk=substr($1,4,6); print $1 " " spk}' > $dir/utt2spk || exit 1;
cat $dir/utt2spk | sort -k 2 | utils/utt2spk_to_spk2utt.pl > $dir/spk2utt || exit 1;

echo Switchboard phase 1 data preparation succeeded.

utils/fix_data_dir.sh $dest
