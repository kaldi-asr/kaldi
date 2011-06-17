# This script should be run from the directory where it is located (i.e. data_prep)
# Copyright 2010-2011 Microsoft Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


# The input is the 3 CDs from the LDC distribution of Resource Management.
# The script's argument is a directory which has three subdirectories:
# rm1_audio1  rm1_audio2  rm2_audio

# Note: when creating your own data preparation scripts, it's a good idea
# to make sure that the speaker id (if present) is a prefix of the utterance
# id, that the output scp file is sorted on utterance id, and that the 
# transcription file is exactly the same length as the scp file and is also
# sorted on utterance id (missing transcriptions should be removed from the
# scp file using e.g. ../scripts/filter_scp.pl)
# You get get some guidance how to deal with channels and segments (not
# an issue in RM) from ../scripts/make_mfcc_train_segs.sh.

if [ $# != 1 ]; then
   echo "Usage: ./run.sh /path/to/RM"
   exit 1; 
fi 

RMROOT=$1
if [ ! -d $RMROOT/rm1_audio1 -o ! -d $RMROOT/rm1_audio2 ]; then
  echo "Error: run.sh requires a directory argument that contains rm1_audio1 and rm1_audio2"
  exit 1; 
fi  

if [ ! -d $RMROOT/rm2_audio ]; then
  echo "**Warning: $RMROOT/rm2_audio does not exist; won't create spk2gender.map file correctly***"
  sleep 1
fi  

(
  find $RMROOT/rm1_audio1/rm1/ind_trn -iname '*.sph';
  find $RMROOT/rm1_audio2/2_4_2/rm1/ind/dev_aug -iname '*.sph';
) | perl -ane ' m:/sa\d.sph:i || m:/sb\d\d.sph:i || print; '  > train_sph.flist



# make_trans.pl also creates the utterance id's and the kaldi-format scp file.
./make_trans.pl trn train_sph.flist $RMROOT/rm1_audio1/rm1/doc/al_sents.snr train_trans.txt train_sph.scp
mv train_trans.txt tmp; sort -k 1 tmp > train_trans.txt
mv train_sph.scp tmp; sort -k 1 tmp > train_sph.scp

sph2pipe=`cd ../../../..; echo $PWD/tools/sph2pipe_v2.5/sph2pipe`
if [ ! -f $sph2pipe ]; then
   echo "Could not find the sph2pipe program at $sph2pipe";
   exit 1;
fi
awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < train_sph.scp > train_wav.scp

cat train_wav.scp | perl -ane 'm/^(\w+_(\w+)\w_\w+) / || die; print "$1 $2\n"' > train.utt2spk
cat train.utt2spk | sort -k 2 | ../scripts/utt2spk_to_spk2utt.pl > train.spk2utt


for ntest in 1_mar87 2_oct87 4_feb89 5_oct89 6_feb91 7_sep92; do
   n=`echo $ntest | cut -d_ -f 1`
   test=`echo $ntest | cut -d_ -f 2`
   root=$RMROOT/rm1_audio2/2_4_2
   for x in `grep -v ';' $root/rm1/doc/tests/$ntest/${n}_indtst.ndx`; do
      echo "$root/$x ";
  done > test_${test}_sph.flist
done

# make_trans.pl also creates the utterance id's and the kaldi-format scp file.
for test in mar87 oct87 feb89 oct89 feb91 sep92; do
  ./make_trans.pl ${test} test_${test}_sph.flist $RMROOT/rm1_audio1/rm1/doc/al_sents.snr test_${test}_trans.txt test_${test}_sph.scp
   mv test_${test}_trans.txt tmp; sort -k 1 tmp > test_${test}_trans.txt
   mv test_${test}_sph.scp tmp; sort -k 1 tmp > test_${test}_sph.scp

  awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < test_${test}_sph.scp  > test_${test}_wav.scp

  cat test_${test}_wav.scp | perl -ane 'm/^(\w+_(\w+)\w_\w+) / || die; print "$1 $2\n"' > test_${test}.utt2spk
  cat test_${test}.utt2spk | sort -k 2 | ../scripts/utt2spk_to_spk2utt.pl > test_${test}.spk2utt
done

cat $RMROOT/rm1_audio2/2_5_1/rm1/doc/al_spkrs.txt \
 $RMROOT/rm2_audio/3-1.2/rm2/doc/al_spkrs.txt | \
 perl -ane 'tr/A-Z/a-z/;print;' | grep -v ';' | \
     awk '{print $1, $2}' | sort | uniq > spk2gender.map

../scripts/make_rm_lm.pl $RMROOT/rm1_audio1/rm1/doc/wp_gram.txt  > G.txt 

# Getting lexicon
../scripts/make_rm_dict.pl  $RMROOT/rm1_audio2/2_4_2/score/src/rdev/pcdsril.txt > lexicon.txt

echo Succeeded.
