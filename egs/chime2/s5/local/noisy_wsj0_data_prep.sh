#!/usr/bin/env bash
set -e

# Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

# This is modified from the script in standard Kaldi recipe to account
# for the way the WSJ data is structured on the Edinburgh systems. 
# - Arnab Ghoshal, 29/05/12

if [ $# -ne 1 ]; then
  printf "\nUSAGE: %s <corpus-directory>\n\n" `basename $0`
  echo "The argument should be a the top-level WSJ corpus directory."
  echo "It is assumed that there will be a 'wsj0' and a 'wsj1' subdirectory"
  echo "within the top-level corpus directory."
  exit 1;
fi

CORPUS=$1

dir=`pwd`/data/local/data
lmdir=`pwd`/data/local/nist_lm
mkdir -p $dir $lmdir
local=`pwd`/local
utils=`pwd`/utils

. ./path.sh # Needed for KALDI_ROOT
export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
  echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
  exit 1;
fi

cd $dir

# reverb list for SI-84

find $1/si_tr_s -name '*.wav' |  sort -u > train_si84_noisy.flist



# Dev-set Hub 1,2 (503, 913 utterances)

# Note: the ???'s below match WSJ and SI_DT, or wsj and si_dt.  
# Sometimes this gets copied from the CD's with upcasing, don't know 
# why (could be older versions of the disks).
find $1/si_dt_20  -name '*.wav' | sort -u > dev_dt_20_noisy.flist
find $1/si_dt_05  -name '*.wav' | sort -u > dev_dt_05_noisy.flist

find $1/si_et_20  -name '*.wav' | sort -u > test_eval92_noisy.flist
find $1/si_et_05  -name '*.wav' | sort -u > test_eval92_5k_noisy.flist


# Finding the transcript files:
#find -L $CORPUS -iname '*.dot' > dot_files.flist
if [ ! -e $dir/dot_files.flist ]; then
  echo "Could not find $dir/dot_files.flist files, first run clean_data_prep.sh";
  exit 1;
fi

# Convert the transcripts into our format (no normalization yet)
# adding suffix to utt_id 
# 1 for reverb condition
for x in train_si84_noisy dev_dt_05_noisy dev_dt_20_noisy test_eval92_noisy test_eval92_5k_noisy; do
  cat $x.flist | perl -e ' 
    while(<>) {
      m:^\S+/(\w+)\.wav$: || die "Bad line $_";
      $id = $1;
      $id =~ tr/A-Z/a-z/;
      print "$id $_"; 
    }
  ' | sort > ${x}_wav_tmp.scp
  #cat ${x}_wav_tmp.scp | awk '{print $1}' \
  #  | $local/find_transcripts.pl dot_files.flist > ${x}_tmp.trans1
  cat ${x}_wav_tmp.scp | perl -e '
    while(<STDIN>) {
      @A=split(" ", $_);
      @B=split("/", $_);
      $abs_path_len=@B;
      $condition=$B[$abs_path_len-5];
      if ($condition eq "9dB") {$key_suffix=2;}
      elsif ($condition eq "6dB") {$key_suffix=3;}
      elsif ($condition eq "3dB") {$key_suffix=4;}
      elsif ($condition eq "0dB") {$key_suffix=5;}
      elsif ($condition eq "m3dB") {$key_suffix=6;}
      elsif ($condition eq "m6dB") {$key_suffix=7;}
      else {print STDERR "error condition $condition";} 
      print $A[0].$key_suffix." ".$A[1]."\n"; 
    }
  ' | sort -k1 > ${x}_wav.scp
  cat ${x}_wav.scp | awk '{print $1}' \
    | $local/find_noisy_transcripts.pl dot_files.flist > ${x}.trans1 
done


# Do some basic normalization steps.  At this point we don't remove OOVs--
# that will be done inside the training scripts, as we'd like to make the
# data-preparation stage independent of the specific lexicon used.
noiseword="<NOISE>";
for x in train_si84_noisy dev_dt_05_noisy dev_dt_20_noisy test_eval92_noisy test_eval92_5k_noisy; do
  cat $x.trans1 | $local/normalize_transcript.pl $noiseword \
    | sort > $x.txt || exit 1;
done
 
# Create scp's with wav's. (the wv1 in the distribution is not really wav, it is sph.)
#for x in train_si84_clean test_eval92_clean test_eval92_5k_clean dev_dt_05_clean dev_dt_20_clean; do
#  awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < ${x}_sph.scp \
#    > ${x}_wav.scp
#done

# Make the utt2spk and spk2utt files.
for x in train_si84_noisy dev_dt_05_noisy dev_dt_20_noisy test_eval92_noisy test_eval92_5k_noisy; do
  cat ${x}_wav.scp | awk '{print $1}' \
    | perl -ane 'chop; m:^...:; print "$_ $&\n";' > $x.utt2spk
  cat $x.utt2spk | $utils/utt2spk_to_spk2utt.pl > $x.spk2utt || exit 1;
done

echo "Data preparation succeeded"
