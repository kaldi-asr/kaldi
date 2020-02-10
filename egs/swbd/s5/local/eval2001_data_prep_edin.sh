#!/usr/bin/env bash

# Hub-5 Eval 2001 data preparation 
# Author:  Arnab Ghoshal (March 2013)

# To be run from one directory above this script.

# The input is a directory name containing the 2001 Hub5 English evaluation 
# speech data (LDC2002S13), and the corresponding STM and GLM files.
# e.g. see
# http://www.ldc.upenn.edu/Catalog/catalogEntry.jsp?catalogId=LDC2002S13
#
# Example usage:
# local/eval2001_data_prep_edin.sh hub5/2001 hub5/2001/ref/stm hub5/2001/ref/glm

if [ $# -ne 3 ]; then
  echo "Usage: "`basename $0`" <speech-dir> <stm-file> <glm-file>"
  echo "See comments in the script for more details"
  exit 1
fi

sdir=$1
stm=$2
glm=$3
[ ! -d $sdir/english ] \
  && echo Expecting directory $sdir/english to be present && exit 1;

. ./path.sh

dir=data/local/eval2001
mkdir -p $dir

find $sdir/english -iname '*.sph' | sort > $dir/sph.flist
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
pem=$sdir/doc/pems/hub5e_01.pem
[ ! -f $pem ] && echo "No such file $pem" && exit 1;
# pem file has lines like: 
# sw4653 A unknown_speaker 191.278009 194.555553

grep -v ';;' $pem \
  | awk '{
           spk=$1"-"$2;
           start=sprintf("%.2f",$4);
           end=sprintf("%.2f", $5);
           utt=sprintf("%s_%06d-%06d",spk,start*100,end*100);
           print utt,spk,start,end;}' \
  | sort -u > $dir/segments

# stm file has lines like:
# sw4653 A sw4653_A 191.278009 194.56 <O,M,P0,P0-M> YEAH WHO ELSE MONKEYS 
# TODO(arnab): We should really be lowercasing this since the Edinburgh
# recipe uses lowercase. This is not used in the actual scoring.
grep -v ';;' $stm \
  | awk '{
           spk=$1"-"$2;
           start=sprintf("%.2f",$4);
           end=sprintf("%.2f", $5);
           utt=sprintf("%s_%06d-%06d",spk,start*100,end*100);
           printf utt; for(n=7;n<=NF;n++) printf(" %s", $n); print ""; }' \
  | sort > $dir/text.all

# We'll use the stm file for sclite scoring.  There seem to be various errors
# in the stm file that upset hubscr.pl, and we fix them here.
sed -e 's:((:(:' -e 's:<B_ASIDE>::g' -e 's:<E_ASIDE>::g' $stm \
  | awk '/^;;/ { print $0 }
         !/^;;/ {
           for(n=1;n<=3;n++) printf("%s ", $n);
           printf("%.2f %.2f", $4, $5);
           for(n=6;n<=NF;n++) printf(" %s", $n); print ""; }' \
  >  $dir/stm
cp $glm $dir/glm

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

dest=data/eval2001
mkdir -p $dest
for x in wav.scp segments text utt2spk spk2utt stm glm reco2file_and_channel; do
  cp $dir/$x $dest/$x
done

echo Data preparation and formatting completed for Eval 2001
echo "(but not MFCC extraction)"

