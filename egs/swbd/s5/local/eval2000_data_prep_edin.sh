#!/bin/bash

# Hub-5 Eval 2000 data preparation 
# Author:  Arnab Ghoshal (Jan 2013)

# To be run from one directory above this script.

# The input is two directory names (possibly the same) ontaining the 
# 2000 Hub5 english evaluation test set and transcripts, which are
# respectively: LDC2002S09  LDC2002T43
# e.g. see
# http://www.ldc.upenn.edu/Catalog/catalogEntry.jsp?catalogId=LDC2002S09
# http://www.ldc.upenn.edu/Catalog/CatalogEntry.jsp?catalogId=LDC2002T43
#
# Example usage:
# local/eval2000_data_prep_edin.sh /exports/work/inf_hcrc_cstr_general/corpora/hub5/2000 /exports/work/inf_hcrc_cstr_general/corpora/hub5/2000/transcr
# The first directory ($sdir) contains the speech data, and the directory
# $sdir/english/ must exist.
# The second directory ($tdir) contains the transcripts, and the directory
# $tdir/reference must exist; in particular we need the file
# $tdir/reference/hub5e00.english.000405.stm

if [ $# -ne 2 ]; then
  echo "Usage: "`basename $0`" <speech-dir> <transcription-dir>"
  echo "See comments in the script for more details"
  exit 1
fi

sdir=$1
tdir=$2
[ ! -d $sdir/english ] \
  && echo Expecting directory $sdir/english to be present && exit 1;
[ ! -d $tdir/reference ] \
  && echo Expecting directory $tdir/reference to be present && exit 1;

. ./path.sh

dir=data/local/eval2000
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
pem=$sdir/english/hub5e_00.pem
[ ! -f $pem ] && echo "No such file $pem" && exit 1;
# pem file has lines like: 
# en_4156 A unknown_speaker 301.85 302.48

grep -v ';;' $pem \
  | awk '{
           spk=$1"-"$2;
           utt=sprintf("%s_%06d-%06d",spk,$4*100,$5*100);
           print utt,spk,$4,$5;}' \
  | sort -u > $dir/segments

# stm file has lines like:
# en_4156 A en_4156_A 357.64 359.64 <O,en,F,en-F>  HE IS A POLICE OFFICER 
# TODO(arnab): We should really be lowercasing this since the Edinburgh
# recipe uses lowercase. This is not used in the actual scoring.
grep -v ';;' $tdir/reference/hub5e00.english.000405.stm \
  | awk '{
           spk=$1"-"$2;
           utt=sprintf("%s_%06d-%06d",spk,$4*100,$5*100);
           printf utt; for(n=7;n<=NF;n++) printf(" %s", $n); print ""; }' \
  | sort > $dir/text.all

# We'll use the stm file for sclite scoring.  There seem to be various errors
# in the stm file that upset hubscr.pl, and we fix them here.
sed -e 's:((:(:' -e 's:<B_ASIDE>::g' -e 's:<E_ASIDE>::g' \
  $tdir/reference/hub5e00.english.000405.stm >  $dir/stm
cp $tdir/reference/en20000405_hub5.glm  $dir/glm

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

dest=data/eval2000
mkdir -p $dest
for x in wav.scp segments text utt2spk spk2utt stm glm reco2file_and_channel; do
  cp $dir/$x $dest/$x
done

echo Data preparation and formatting completed for Eval 2000
echo "(but not MFCC extraction)"

