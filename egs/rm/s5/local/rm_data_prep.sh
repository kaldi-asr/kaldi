#!/bin/bash
#
# Copyright 2010-2012 Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# To be run from one directory above this script.

# The input is the 3 CDs from the LDC distribution of Resource Management.
# The script's argument is a directory which has three subdirectories:
# rm1_audio1  rm1_audio2  rm2_audio

# Note: when creating your own data preparation scripts, it's a good idea
# to make sure that the speaker id (if present) is a prefix of the utterance
# id, that the output scp file is sorted on utterance id, and that the 
# transcription file is exactly the same length as the scp file and is also
# sorted on utterance id (missing transcriptions should be removed from the
# scp file using e.g. scripts/filter_scp.pl)

if [ $# != 1 ]; then
  echo "Usage: ../../local/RM_data_prep.sh /path/to/RM"
  exit 1; 
fi 

export LC_ALL=C

RMROOT=$1

tmpdir=data/local/tmp
mkdir -p $tmpdir
. ./path.sh || exit 1; # for KALDI_ROOT

if [ ! -d $RMROOT/rm1_audio1 -o ! -d $RMROOT/rm1_audio2 ]; then
   echo "Error: rm_data_prep.sh requires a directory argument (an absolute pathname) that contains rm1_audio1 and rm1_audio2"
   exit 1; 
fi  

if [ ! -d $RMROOT/rm2_audio ]; then
   echo "**Warning: $RMROOT/rm2_audio does not exist; won't create spk2gender file correctly***"
   sleep 1
fi  

(
    find $RMROOT/rm1_audio1/rm1/ind_trn -iname '*.sph';
    find $RMROOT/rm1_audio2/2_4_2/rm1/ind/dev_aug -iname '*.sph';
) | perl -ane ' m:/sa\d.sph:i || m:/sb\d\d.sph:i || print; '  > $tmpdir/train_sph.flist


dir=data/train
mkdir -p $dir

# make_trans.pl also creates the utterance id's and the kaldi-format scp file.
local/make_trans.pl trn $tmpdir/train_sph.flist $RMROOT/rm1_audio1/rm1/doc/al_sents.snr >(sort -k1 >$dir/text) \
  >(sort -k1 >$dir/sph.scp)


sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
[ ! -f $sph2pipe ] && echo "Could not find the sph2pipe program at $sph2pipe" && exit 1;

awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' <$dir/sph.scp > $dir/wav.scp
rm $dir/sph.scp

cat $dir/wav.scp | perl -ane 'm/^((\w+)\w_\w+_\w+) / || die; print "$1 $2\n"' > $dir/utt2spk
utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt


for ntest in 1_mar87 2_oct87 4_feb89 5_oct89 6_feb91 7_sep92; do
  n=`echo $ntest | cut -d_ -f 1` # e.g. n = 1, 2, 4, 5..
  test=`echo $ntest | cut -d_ -f 2` # e.g. test=mar87, oct87...
  dir=data/test_${test}
  mkdir -p $dir
  root=$RMROOT/rm1_audio2/2_4_2
  for x in `grep -v ';' $root/rm1/doc/tests/$ntest/${n}_indtst.ndx`; do
    echo "$root/$x ";
  done | sort > $dir/sph.flist

  local/make_trans.pl ${test} $dir/sph.flist $RMROOT/rm1_audio1/rm1/doc/al_sents.snr \
     >(sort -k1 >$dir/text) >(sort -k1 >$dir/sph.scp)
  sleep 0.25 # At one point I had the next line failing because $dir/sph.scp appeared not
             # to exist.  Adding this sleep statement appeared to fix the problem.
  awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' <$dir/sph.scp >$dir/wav.scp
  rm $dir/sph.flist $dir/sph.scp

  cat $dir/wav.scp | perl -ane 'm/^((\w+)\w_\w+_\w+) / || die; print "$1 $2\n"' > $dir/utt2spk
  utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
done

cat $RMROOT/rm1_audio2/2_5_1/rm1/doc/al_spkrs.txt \
    $RMROOT/rm2_audio/3-1.2/rm2/doc/al_spkrs.txt | \
    perl -ane 'tr/A-Z/a-z/;print;' | grep -v ';' | \
    awk '{print $1, $2}' | sort | uniq > $tmpdir/spk2gender || exit 1;

for t in train test_mar87 test_oct87 test_feb89 test_oct89 test_feb91 test_sep92; do
  utils/filter_scp.pl data/$t/spk2utt $tmpdir/spk2gender >data/$t/spk2gender
done

local/make_rm_lm.pl $RMROOT/rm1_audio1/rm1/doc/wp_gram.txt  > $tmpdir/G.txt || exit 1;

mkdir data/local/dict

# Getting lexicon
local/make_rm_dict.pl  $RMROOT/rm1_audio2/2_4_2/score/src/rdev/pcdsril.txt \
   > data/local/dict/lexicon.txt || exit 1;

# Get phone lists...
grep -v -w sil data/local/dict/lexicon.txt | \
  awk '{for(n=2;n<=NF;n++) { p[$n]=1; }} END{for(x in p) {print x}}' | sort > data/local/dict/nonsilence_phones.txt
echo sil > data/local/dict/silence_phones.txt
echo sil > data/local/dict/optional_silence.txt
touch data/local/dict/extra_questions.txt # no extra questions, as we have no stress or tone markers.

echo RM_data_prep succeeded.
