#!/bin/bash
set -e

# Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

if [ $# -ne 2 ]; then
  printf "\nUSAGE: %s <corpus-directory>\n\n" `basename $0`
  echo "The argument should be a the top-level WSJ corpus directory."
  echo "It is assumed that there will be a 'wsj0' and a 'wsj1' subdirectory"
  echo "within the top-level corpus directory."
  exit 1;
fi

AURORA=$1
CORPUS=$2

dir=`pwd`/data/local/data
lmdir=`pwd`/data/local/nist_lm
mkdir -p $dir $lmdir
local=`pwd`/local
utils=`pwd`/utils

. ./path.sh # Needed for KALDI_ROOT
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
  echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
  exit 1;
fi

if [ -z $IRSTLM ] ; then
  export IRSTLM=$KALDI_ROOT/tools/irstlm/
fi
export PATH=${PATH}:$IRSTLM/bin
if ! command -v prune-lm >/dev/null 2>&1 ; then
  echo "$0: Error: the IRSTLM is not available or compiled" >&2
  echo "$0: Error: We used to install it by default, but." >&2
  echo "$0: Error: this is no longer the case." >&2
  echo "$0: Error: To install it, go to $KALDI_ROOT/tools" >&2
  echo "$0: Error: and run extras/install_irstlm.sh" >&2
  exit 1
fi

cd $dir

# SI-84 clean training data
cat $AURORA/lists/training_clean_sennh_16k.list \
  | $local/aurora2flist.pl $AURORA | sort -u > train_si84_clean.flist

# SI-84 multi-condition training data
cat $AURORA/lists/training_multicondition_16k.list \
  | $local/aurora2flist.pl $AURORA | sort -u > train_si84_multi.flist

#Dev Set
for x in $(seq -f "%02g" 01 14); do
  # Dev-set 1 (330x14 utterances)
  cat $AURORA/lists/devtest${x}_0330_16k.list | perl -e ' 
    while(<STDIN>) {
      @A=split("/", $_);        
      @B=split("_", $A[0]);
      print $B[0].$B[1]."_".$B[2]."/".$_;  
    }
  ' | $local/aurora2flist.pl $AURORA | sort -u > dev_0330_${x}.flist
  # Dev-set 2 (1206x14 utterances)
  cat $AURORA/lists/devtest${x}_1206_16k.list | perl -e '
    while(<STDIN>) {
      @A=split("/", $_);        
      @B=split("_", $A[0]);
      print $B[0].$B[1]."_".$B[2]."/".$_;  
    }
  ' | $local/aurora2flist.pl $AURORA | sort -u > dev_1206_${x}.flist
done

#Test Set
for x in $(seq -f "%02g" 01 14); do
  # test set 1 (166x14 utterances)
  cat $AURORA/lists/test${x}_0166_16k.list \
  | $local/aurora2flist.pl $AURORA | sort -u > test_0166_${x}.flist 
  cat $AURORA/lists/test${x}_0330_16k.list \
  | $local/aurora2flist.pl $AURORA | sort -u > test_eval92_${x}.flist
done

# Finding the transcript files:
find -L $CORPUS -iname '*.dot' > dot_files.flist

# Convert the transcripts into our format (no normalization yet)
# adding suffix to utt_id 
# 0 for clean condition

# Trans and sph for Train Set
x=train_si84_clean
$local/flist2scp_12.pl $x.flist | sort > ${x}_sph_tmp.scp
cat ${x}_sph_tmp.scp | awk '{print $1}' \
  | $local/find_transcripts.pl dot_files.flist > ${x}_tmp.trans1
cat ${x}_sph_tmp.scp | awk '{printf("%s0 %s\n", $1, $2);}' > ${x}_sph.scp
cat ${x}_tmp.trans1 | awk '{printf("%s0 ", $1); for(i=2;i<=NF;i++) printf("%s ", $i); printf("\n");}' > ${x}.trans1

x=train_si84_multi
$local/flist2scp_12.pl $x.flist | sort > ${x}_sph_tmp.scp
cat ${x}_sph_tmp.scp | awk '{print $1}' \
  | $local/find_transcripts.pl dot_files.flist > ${x}_tmp.trans1
cat ${x}_sph_tmp.scp | awk '{printf("%s1 %s\n", $1, $2);}' | grep -v '408o0302\.wv2$'> ${x}_sph.scp
cat ${x}_tmp.trans1 | awk '{printf("%s1 ", $1); for(i=2;i<=NF;i++) printf("%s ", $i); printf("\n");}' \
  | sort -u > ${x}.trans1

# Trans and sph for Dev Set
for x in $(seq -f "%02g" 01 14); do
  $local/flist2scp_12.pl dev_0330_${x}.flist | sort > dev_0330_${x}_sph_tmp.scp
  $local/flist2scp_12.pl dev_1206_${x}.flist | sort > dev_1206_${x}_sph_tmp.scp
  cat dev_0330_${x}_sph_tmp.scp | awk '{print $1}' \
    | $local/find_transcripts.pl dot_files.flist > dev_0330_${x}_tmp.trans1
  cat dev_1206_${x}_sph_tmp.scp | awk '{print $1}' \
    | $local/find_transcripts.pl dot_files.flist > dev_1206_${x}_tmp.trans1
  cat dev_0330_${x}_sph_tmp.scp | perl -e \
  ' $condition="$ARGV[0]";
    if ($condition eq "01") {$suffix=0;}
    elsif ($condition eq "02") {$suffix=1;} 
    elsif ($condition eq "03") {$suffix=2;} 
    elsif ($condition eq "04") {$suffix=3;} 
    elsif ($condition eq "05") {$suffix=4;} 
    elsif ($condition eq "06") {$suffix=5;} 
    elsif ($condition eq "07") {$suffix=6;} 
    elsif ($condition eq "08") {$suffix=7;} 
    elsif ($condition eq "09") {$suffix=8;} 
    elsif ($condition eq "10") {$suffix=9;} 
    elsif ($condition eq "11") {$suffix=a;} 
    elsif ($condition eq "12") {$suffix=b;} 
    elsif ($condition eq "13") {$suffix=c;} 
    elsif ($condition eq "14") {$suffix=d;} 
    else {print STDERR "error condition $condition\n";}
    while(<STDIN>) {
      @A=split(" ", $_);  
      print $A[0].$suffix." ".$A[1]."\n"; 
    }
  ' $x > dev_0330_${x}_sph.scp 
  
  cat dev_1206_${x}_sph_tmp.scp | perl -e \
  ' $condition="$ARGV[0]";
    if ($condition eq "01") {$suffix=0;}
    elsif ($condition eq "02") {$suffix=1;} 
    elsif ($condition eq "03") {$suffix=2;} 
    elsif ($condition eq "04") {$suffix=3;} 
    elsif ($condition eq "05") {$suffix=4;} 
    elsif ($condition eq "06") {$suffix=5;} 
    elsif ($condition eq "07") {$suffix=6;} 
    elsif ($condition eq "08") {$suffix=7;} 
    elsif ($condition eq "09") {$suffix=8;} 
    elsif ($condition eq "10") {$suffix=9;} 
    elsif ($condition eq "11") {$suffix=a;} 
    elsif ($condition eq "12") {$suffix=b;} 
    elsif ($condition eq "13") {$suffix=c;} 
    elsif ($condition eq "14") {$suffix=d;} 
    else {print STDERR "error condition $condition";}
    while(<STDIN>) {
      @A=split(" ", $_);  
      print $A[0].$suffix." ".$A[1]."\n";
    }
  ' $x > dev_1206_${x}_sph.scp
  
  cat dev_0330_${x}_tmp.trans1 | perl -e \
  ' $condition="$ARGV[0]";
    if ($condition eq "01") {$suffix=0;}
    elsif ($condition eq "02") {$suffix=1;} 
    elsif ($condition eq "03") {$suffix=2;} 
    elsif ($condition eq "04") {$suffix=3;} 
    elsif ($condition eq "05") {$suffix=4;} 
    elsif ($condition eq "06") {$suffix=5;} 
    elsif ($condition eq "07") {$suffix=6;} 
    elsif ($condition eq "08") {$suffix=7;} 
    elsif ($condition eq "09") {$suffix=8;} 
    elsif ($condition eq "10") {$suffix=9;} 
    elsif ($condition eq "11") {$suffix=a;} 
    elsif ($condition eq "12") {$suffix=b;} 
    elsif ($condition eq "13") {$suffix=c;} 
    elsif ($condition eq "14") {$suffix=d;} 
    else {print STDERR "error condition $condition";}
    while(<STDIN>) {
      @A=split(" ", $_);  
      print $A[0].$suffix;
      for ($i=1; $i < @A; $i++) {print " ".$A[$i];}
      print "\n";
    }
  ' $x > dev_0330_${x}.trans1
 
  cat dev_1206_${x}_tmp.trans1 | perl -e \
  ' $condition="$ARGV[0]";
    if ($condition eq "01") {$suffix=0;}
    elsif ($condition eq "02") {$suffix=1;} 
    elsif ($condition eq "03") {$suffix=2;} 
    elsif ($condition eq "04") {$suffix=3;} 
    elsif ($condition eq "05") {$suffix=4;} 
    elsif ($condition eq "06") {$suffix=5;} 
    elsif ($condition eq "07") {$suffix=6;} 
    elsif ($condition eq "08") {$suffix=7;} 
    elsif ($condition eq "09") {$suffix=8;} 
    elsif ($condition eq "10") {$suffix=9;} 
    elsif ($condition eq "11") {$suffix=a;} 
    elsif ($condition eq "12") {$suffix=b;} 
    elsif ($condition eq "13") {$suffix=c;} 
    elsif ($condition eq "14") {$suffix=d;} 
    else {print STDERR "error condition $condition";}
    while(<STDIN>) {
      @A=split(" ", $_);  
      print $A[0].$suffix;
      for ($i=1; $i < @A; $i++) {print " ".$A[$i];}
      print "\n";
    }
  ' $x > dev_1206_${x}.trans1

done

cat dev_0330_*_sph.scp | sort -k1 > dev_0330_sph.scp 
cat dev_1206_*_sph.scp | sort -k1 > dev_1206_sph.scp
cat dev_0330_??.trans1 | sort -k1 > dev_0330.trans1
cat dev_1206_??.trans1 | sort -k1 > dev_1206.trans1


# Trans and sph for Test Set
for x in $(seq -f "%02g" 01 14); do
  $local/flist2scp_12.pl test_0166_${x}.flist | sort > test_0166_${x}_sph_tmp.scp
  $local/flist2scp_12.pl test_eval92_${x}.flist | sort > test_eval92_${x}_sph_tmp.scp
  cat test_0166_${x}_sph_tmp.scp | awk '{print $1}' \
    | $local/find_transcripts.pl dot_files.flist > test_0166_${x}_tmp.trans1
  cat test_eval92_${x}_sph_tmp.scp | awk '{print $1}' \
    | $local/find_transcripts.pl dot_files.flist > test_eval92_${x}_tmp.trans1
  cat test_0166_${x}_sph_tmp.scp | perl -e \
  ' $condition="$ARGV[0]";
    if ($condition eq "01") {$suffix=0;}
    elsif ($condition eq "02") {$suffix=1;} 
    elsif ($condition eq "03") {$suffix=2;} 
    elsif ($condition eq "04") {$suffix=3;} 
    elsif ($condition eq "05") {$suffix=4;} 
    elsif ($condition eq "06") {$suffix=5;} 
    elsif ($condition eq "07") {$suffix=6;} 
    elsif ($condition eq "08") {$suffix=7;} 
    elsif ($condition eq "09") {$suffix=8;} 
    elsif ($condition eq "10") {$suffix=9;} 
    elsif ($condition eq "11") {$suffix=a;} 
    elsif ($condition eq "12") {$suffix=b;} 
    elsif ($condition eq "13") {$suffix=c;} 
    elsif ($condition eq "14") {$suffix=d;} 
    else {print STDERR "error condition $condition";}
    while(<STDIN>) {
      @A=split(" ", $_);  
      print $A[0].$suffix." ".$A[1]."\n";
    }
  ' $x > test_0166_${x}_sph.scp 
  
  cat test_eval92_${x}_sph_tmp.scp | perl -e \
  ' $condition="$ARGV[0]";
    if ($condition eq "01") {$suffix=0;}
    elsif ($condition eq "02") {$suffix=1;} 
    elsif ($condition eq "03") {$suffix=2;} 
    elsif ($condition eq "04") {$suffix=3;} 
    elsif ($condition eq "05") {$suffix=4;} 
    elsif ($condition eq "06") {$suffix=5;} 
    elsif ($condition eq "07") {$suffix=6;} 
    elsif ($condition eq "08") {$suffix=7;} 
    elsif ($condition eq "09") {$suffix=8;} 
    elsif ($condition eq "10") {$suffix=9;} 
    elsif ($condition eq "11") {$suffix=a;} 
    elsif ($condition eq "12") {$suffix=b;} 
    elsif ($condition eq "13") {$suffix=c;} 
    elsif ($condition eq "14") {$suffix=d;} 
    else {print STDERR "error condition $condition";}
    while(<STDIN>) {
      @A=split(" ", $_);  
      print $A[0].$suffix." ".$A[1]."\n";
    }
  ' $x > test_eval92_${x}_sph.scp
  
  cat test_0166_${x}_tmp.trans1 | perl -e \
  ' $condition="$ARGV[0]";
    if ($condition eq "01") {$suffix=0;}
    elsif ($condition eq "02") {$suffix=1;} 
    elsif ($condition eq "03") {$suffix=2;} 
    elsif ($condition eq "04") {$suffix=3;} 
    elsif ($condition eq "05") {$suffix=4;} 
    elsif ($condition eq "06") {$suffix=5;} 
    elsif ($condition eq "07") {$suffix=6;} 
    elsif ($condition eq "08") {$suffix=7;} 
    elsif ($condition eq "09") {$suffix=8;} 
    elsif ($condition eq "10") {$suffix=9;} 
    elsif ($condition eq "11") {$suffix=a;} 
    elsif ($condition eq "12") {$suffix=b;} 
    elsif ($condition eq "13") {$suffix=c;} 
    elsif ($condition eq "14") {$suffix=d;} 
    else {print STDERR "error condition $condition";}
    while(<STDIN>) {
      @A=split(" ", $_);  
      print $A[0].$suffix;
      for ($i=1; $i < @A; $i++) {print " ".$A[$i];}
      print "\n";
    }
  ' $x > test_0166_${x}.trans1
 
  cat test_eval92_${x}_tmp.trans1 | perl -e \
  ' $condition="$ARGV[0]";
    if ($condition eq "01") {$suffix=0;}
    elsif ($condition eq "02") {$suffix=1;} 
    elsif ($condition eq "03") {$suffix=2;} 
    elsif ($condition eq "04") {$suffix=3;} 
    elsif ($condition eq "05") {$suffix=4;} 
    elsif ($condition eq "06") {$suffix=5;} 
    elsif ($condition eq "07") {$suffix=6;} 
    elsif ($condition eq "08") {$suffix=7;} 
    elsif ($condition eq "09") {$suffix=8;} 
    elsif ($condition eq "10") {$suffix=9;} 
    elsif ($condition eq "11") {$suffix=a;} 
    elsif ($condition eq "12") {$suffix=b;} 
    elsif ($condition eq "13") {$suffix=c;} 
    elsif ($condition eq "14") {$suffix=d;} 
    else {print STDERR "error condition $condition";}
    while(<STDIN>) {
      @A=split(" ", $_);  
      print $A[0].$suffix;
      for ($i=1; $i < @A; $i++) {print " ".$A[$i];}
      print "\n";
    }
  ' $x > test_eval92_${x}.trans1

done

cat test_0166_*_sph.scp | sort -k1 > test_0166_sph.scp 
cat test_eval92_*_sph.scp | sort -k1 > test_eval92_sph.scp
cat test_0166_??.trans1 | sort -k1 > test_0166.trans1
cat test_eval92_??.trans1 | sort -k1 > test_eval92.trans1


# Do some basic normalization steps.  At this point we don't remove OOVs--
# that will be done inside the training scripts, as we'd like to make the
# data-preparation stage independent of the specific lexicon used.
noiseword="<NOISE>";
for x in train_si84_clean train_si84_multi test_eval92 test_0166 dev_0330 dev_1206; do
  cat $x.trans1 | $local/normalize_transcript.pl $noiseword \
    | sort > $x.txt || exit 1;
done

# Create scp's with wav's. (the wv1 in the distribution is not really wav, it is sph.)
for x in train_si84_clean train_si84_multi test_eval92 test_0166 dev_0330 dev_1206; do
  awk '{printf("%s sox -B -r 16k -e signed -b 16 -c 1 -t raw %s -t wav - |\n", $1, $2);}' < ${x}_sph.scp \
    > ${x}_wav.scp
done

# Make the utt2spk and spk2utt files.
for x in train_si84_clean train_si84_multi test_eval92 test_0166 dev_0330 dev_1206; do
  cat ${x}_sph.scp | awk '{print $1}' \
    | perl -ane 'chop; m:^...:; print "$_ $&\n";' > $x.utt2spk
  cat $x.utt2spk | $utils/utt2spk_to_spk2utt.pl > $x.spk2utt || exit 1;
done

#in case we want to limit lm's on most frequent words, copy lm training word frequency list
cp $CORPUS/11-13.1/wsj0/doc/lng_modl/vocab/wfl_64.lst $lmdir
chmod u+w $lmdir/*.lst # had weird permissions on source.

# The 20K vocab, open-vocabulary language model (i.e. the one with UNK), without
# verbalized pronunciations.   This is the most common test setup, I understand.

cp $CORPUS/11-13.1/wsj0/doc/lng_modl/base_lm/bcb20onp.z $lmdir/lm_bg.arpa.gz || exit 1;
chmod u+w $lmdir/lm_bg.arpa.gz

# trigram would be:
cat $CORPUS/11-13.1/wsj0/doc/lng_modl/base_lm/tcb20onp.z | \
  perl -e 'while(<>){ if(m/^\\data\\/){ print; last;  } } while(<>){ print; }' \
  | gzip -c -f > $lmdir/lm_tg.arpa.gz || exit 1;

prune-lm --threshold=1e-7 $lmdir/lm_tg.arpa.gz $lmdir/lm_tgpr.arpa || exit 1;
gzip -f $lmdir/lm_tgpr.arpa || exit 1;

# repeat for 5k language models
cp $CORPUS/11-13.1/wsj0/doc/lng_modl/base_lm/bcb05onp.z  $lmdir/lm_bg_5k.arpa.gz || exit 1;
chmod u+w $lmdir/lm_bg_5k.arpa.gz

# trigram would be: !only closed vocabulary here!
cp $CORPUS/11-13.1/wsj0/doc/lng_modl/base_lm/tcb05cnp.z $lmdir/lm_tg_5k.arpa.gz || exit 1;
chmod u+w $lmdir/lm_tg_5k.arpa.gz
gunzip $lmdir/lm_tg_5k.arpa.gz
tail -n 4328839 $lmdir/lm_tg_5k.arpa | gzip -c -f > $lmdir/lm_tg_5k.arpa.gz
rm $lmdir/lm_tg_5k.arpa

prune-lm --threshold=1e-7 $lmdir/lm_tg_5k.arpa.gz $lmdir/lm_tgpr_5k.arpa || exit 1;
gzip -f $lmdir/lm_tgpr_5k.arpa || exit 1;


if [ ! -f wsj0-train-spkrinfo.txt ] || [ `cat wsj0-train-spkrinfo.txt | wc -l` -ne 134 ]; then
  rm -f wsj0-train-spkrinfo.txt
  wget http://www.ldc.upenn.edu/Catalog/docs/LDC93S6A/wsj0-train-spkrinfo.txt \
    || ( echo "Getting wsj0-train-spkrinfo.txt from backup location" && \
         wget --no-check-certificate https://sourceforge.net/projects/kaldi/files/wsj0-train-spkrinfo.txt );
fi

if [ ! -f wsj0-train-spkrinfo.txt ]; then
  echo "Could not get the spkrinfo.txt file from LDC website (moved)?"
  echo "This is possibly omitted from the training disks; couldn't find it." 
  echo "Everything else may have worked; we just may be missing gender info"
  echo "which is only needed for VTLN-related diagnostics anyway."
  exit 1
fi
# Note: wsj0-train-spkrinfo.txt doesn't seem to be on the disks but the
# LDC put it on the web.  Perhaps it was accidentally omitted from the
# disks.  

cat $CORPUS/11-13.1/wsj0/doc/spkrinfo.txt \
    ./wsj0-train-spkrinfo.txt  | \
    perl -ane 'tr/A-Z/a-z/; m/^;/ || print;' | \
    awk '{print $1, $2}' | grep -v -- -- | sort | uniq > spk2gender


echo "Data preparation succeeded"
