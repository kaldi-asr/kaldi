#!/usr/bin/env bash

# Hub-5 Eval 1997 data preparation
# Author:  Arnab Ghoshal (Jan 2013)

# To be run from one directory above this script.

# The input is a directory name containing the 1997 Hub5 english evaluation
# test set and transcripts, which is LDC2002S10
# e.g. see
# http://www.ldc.upenn.edu/Catalog/CatalogEntry.jsp?catalogId=LDC2002S10
#
# It is assumed that the transcripts are in a subdirectory called transcr
# However, we download the STM from NIST site:
# ftp://jaguar.ncsl.nist.gov/lvcsr/mar97/eval/hub5e97.english.980618.stm

if [ $# -ne 1 ]; then
  echo "Usage: "`basename $0`" <speech-dir>"
  echo "See comments in the script for more details"
  exit 1
fi

sdir=$1
[ ! -d $sdir/speech ] \
  && echo Expecting directory $sdir/speech to be present && exit 1;
[ ! -d $sdir/transcr ] \
  && echo Expecting directory $sdir/transcr to be present && exit 1;

. ./path.sh

dir=data/local/eval1997
mkdir -p $dir

find $sdir/speech -iname '*.sph' | sort > $dir/sph.flist
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
pem=$sdir/speech/97_hub5e.pem
[ ! -f $pem ] && echo "$0: No such file $pem" && exit 1;
# pem file has lines like:
# en_4156 A unknown_speaker 301.85 302.48
# There is one line in the 97_hub5e.pem with an extra : on the channel
# sw_10022 B: unknown_speaker 281.21 284.37 -- the : is removed
# There are two other mistakes in the pem that are also corrected.
grep -v ';;' $pem | sed -e 's?:??g' \
  | awk '{
           spk=$1"-"$2;  start=$4;  end=$5;
           if (spk == "en_4763-A" && start == 389.14) end=389.40;
           if (spk == "en_5153-A" && start == 593.84) end=594.31;
           utt=sprintf("%s_%06d-%06d", spk, start*100, end*100);
           printf "%s %s %.2f %.2f\n", utt, spk, start, end; }' \
  | sort -u > $dir/segments


# Download the STM and GLM files:
( cd $dir
  rm -f stm glm
  [ -f hub5e97.english.980618.stm ] || \
    wget ftp://jaguar.ncsl.nist.gov/lvcsr/mar97/eval/hub5e97.english.980618.stm
  ln -s hub5e97.english.980618.stm stm
  [ -f en20010117_hub5.glm ] || \
  wget ftp://jaguar.ncsl.nist.gov/rt/rt02/software/en20010117_hub5.glm
  ln -s en20010117_hub5.glm glm
)


# stm file has lines like:
# en_4042 A en_4042_A 227.71 232.26 <O>  BEANS RIGHT THAT IS WHY I SAID BEANS
# One of the segments (sw_10022-B_028120-028437) is removed since it is not
# scored and does not show up in the pem file.
grep -v ';;' $dir/hub5e97.english.980618.stm \
  | awk '{
           spk=$1"-"$2;
           utt=sprintf("%s_%06d-%06d",spk,$4*100,$5*100);
           printf utt; for(n=7;n<=NF;n++) printf(" %s", $n); print ""; }' \
  | sort -k1,1 -u > $dir/text.all
grep -v IGNORE_TIME_SEGMENT_ $dir/text.all > $dir/text

# next line uses command substitution
# Just checking that the segments are the same in pem vs. stm.
! cmp <(awk '{print $1}' $dir/text.all) <(awk '{print $1}' $dir/segments) && \
   echo "Segments from pem file and stm file do not match." && exit 1;

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

dest=data/eval1997
mkdir -p $dest
for x in wav.scp segments text utt2spk spk2utt stm glm reco2file_and_channel; do
  cp $dir/$x $dest/$x
done

echo Data preparation and formatting completed for Eval 2000
echo "(but not MFCC extraction)"
