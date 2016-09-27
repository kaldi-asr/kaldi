#!/bin/bash

###########################################################################################
# This script was copied from egs/fisher_swbd/s5/local/rt03_data_prep.sh
# The source commit was e69198c3dc5633f98eb88e1cdf20b2521a598f21
# Changes made:
#  - Specified path to path.sh
#  - Modified paths to match multi_en naming conventions
###########################################################################################

# RT-03 data preparation (conversational telephone speech part only) 
# Adapted from Arnab Ghoshal's script for Hub-5 Eval 2000 by Peng Qi

# To be run from one directory above this script.

# Expects the standard directory layout for RT-03

if [ $# -ne 1 ]; then
  echo "Usage: "`basename $0`" <rt03-dir>"
  echo "See comments in the script for more details"
  exit 1
fi

sdir=$1
[ ! -d $sdir/data/audio/eval03/english/cts ] \
  && echo Expecting directory $sdir/data/audio/eval03/english/cts to be present && exit 1;
[ ! -d $sdir/data/references/eval03/english/cts ] \
  && echo Expecting directory $tdir/data/references/eval03/english/cts to be present && exit 1;

. ./path.sh

dir=data/local/rt03
mkdir -p $dir

rtroot=$sdir
tdir=$sdir/data/references/eval03/english/cts
sdir=$sdir/data/audio/eval03/english/cts

find $sdir -iname '*.sph' | sort > $dir/sph.flist
sed -e 's?.*/??' -e 's?.sph??' $dir/sph.flist | paste - $dir/sph.flist \
  > $dir/sph.scp

sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
[ ! -x $sph2pipe ] \
  && echo "Could not execute the sph2pipe program at $sph2pipe" && exit 1;

awk -v sph2pipe=$sph2pipe '{
  printf("%s-A %s -f wav -p -c 1 %s |\n", $1, sph2pipe, $2); 
  printf("%s-B %s -f wav -p -c 2 %s |\n", $1, sph2pipe, $2);
}' < $dir/sph.scp | sort > $dir/wav.scp || exit 1;
#side A - channel 1, side B - channel 2

# Get segments file...
# segments file format is: utt-id side-id start-time end-time, e.g.:
# sw02001-A_000098-001156 sw02001-A 0.98 11.56
#pem=$sdir/english/hub5e_00.pem
#[ ! -f $pem ] && echo "No such file $pem" && exit 1;
# pem file has lines like: 
# en_4156 A unknown_speaker 301.85 302.48

#grep -v ';;' $pem \
cat $tdir/*.stm | grep -v ';;' | grep -v inter_segment_gap \
  | awk '{
           spk=$1"-"(($2==1)?"A":"B");
           utt=sprintf("%s_%06d-%06d",spk,$4*100,$5*100);
           print utt,spk,$4,$5;}' \
  | sort -u > $dir/segments

# stm file has lines like:
# en_4156 A en_4156_A 357.64 359.64 <O,en,F,en-F>  HE IS A POLICE OFFICER 
# TODO(arnab): We should really be lowercasing this since the Edinburgh
# recipe uses lowercase. This is not used in the actual scoring.
#grep -v ';;' $tdir/reference/hub5e00.english.000405.stm \
cat $tdir/*.stm | grep -v ';;' | grep -v inter_segment_gap \
  | awk '{
           spk=$1"-"(($2==1)?"A":"B");
           utt=sprintf("%s_%06d-%06d",spk,$4*100,$5*100);
           printf utt; for(n=7;n<=NF;n++) printf(" %s", $n); print ""; }' \
  | sort > $dir/text.all

# We'll use the stm file for sclite scoring.  There seem to be various errors
# in the stm file that upset hubscr.pl, and we fix them here.
cat $tdir/*.stm | \
  sed -e 's:((:(:' -e 's:<B_ASIDE>::g' -e 's:<E_ASIDE>::g' | \
  grep -v inter_segment_gap | \
  awk '{
           printf $1; if ($1==";;") printf(" %s",$2); else printf(($2==1)?" A":" B"); for(n=3;n<=NF;n++) printf(" %s", $n); print ""; }'\
  > $dir/stm  
#$tdir/reference/hub5e00.english.000405.stm >  $dir/stm
cp $rtroot/data/trans_rules/en20030506.glm  $dir/glm

# next line uses command substitution
# Just checking that the segments are the same in pem vs. stm.
! cmp <(awk '{print $1}' $dir/text.all) <(awk '{print $1}' $dir/segments) && \
   echo "Segments from pem file and stm file do not match." && exit 1;

grep -v IGNORE_TIME_SEGMENT_ $dir/text.all > $dir/text
   
# create an utt2spk file that assumes each conversation side is
# a separate speaker.
awk '{print $1,$2;}' $dir/segments > $dir/utt2spk  
utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt

# cp $dir/segments $dir/segments.tmp
# awk '{x=$3-0.05; if (x<0.0) x=0.0; y=$4+0.05; print $1, $2, x, y; }' \
#   $dir/segments.tmp > $dir/segments

awk '{print $1}' $dir/wav.scp \
  | perl -ane '$_ =~ m:^(\S+)-([AB])$: || die "bad label $_";
               print "$1-$2 $1 $2\n"; ' \
  > $dir/reco2file_and_channel || exit 1;

dest=data/rt03/test
mkdir -p $dest
for x in wav.scp segments text utt2spk spk2utt stm glm reco2file_and_channel; do
  cp $dir/$x $dest/$x
done

echo Data preparation and formatting completed for RT-03
echo "(but not MFCC extraction)"

