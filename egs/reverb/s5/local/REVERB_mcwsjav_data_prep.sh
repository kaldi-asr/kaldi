#!/bin/bash

# Copyright 2013 MERL (author: Felix Weninger)
# Contains some code by Microsoft Corporation, Johns Hopkins University (author: Daniel Povey)

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


# for REVERB challenge:

dir=`pwd`/data/local/data
lmdir=`pwd`/data/local/nist_lm
mkdir -p $dir $lmdir
local=`pwd`/local
utils=`pwd`/utils
root=`pwd`

. ./path.sh # Needed for KALDI_ROOT
export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
   exit 1;
fi

cd $dir

MIC=primary

# input corpus (original or processed, tr or dt, etc.)
RWSJ=$1
if [ ! -d "$RWSJ" ]; then
    echo Could not find directory $RWSJ! Check pathnames in corpus.sh!
    exit 1
fi

mcwsjav_mlf=$RWSJ/mlf/WSJ.mlf
if [ ! -z "$4" ]; then
    mcwsjav_mlf=$4
fi

# the name of the dataset to be created
dataset=REVERB_Real_dt

# the WSJCAM0 set that the set is based on (tr, dt, ...)
# this will be used to find the correct transcriptions etc.
dt_or_x=dt

if [ ! -z "$2" ]; then
   dataset=$2
fi
# dt or et
if [ ! -z "$3" ]; then
   dt_or_x=$3
fi

# unfortunately, we need a pointer to HTK baseline
# since the corpus does NOT contain the data set descriptions
# for the REVERB Challenge

taskFileDir=$dir/../reverb_tools/ReleasePackage/reverb_tools_for_asr_ver2.0/taskFiles/1ch
#taskFiles=`ls $taskFileDir/*Data_dt_for_*`
taskFiles=`ls $taskFileDir/RealData_${dt_or_x}_for_1ch_{far,near}*`

dir2=$dir/$dataset
mkdir -p $dir2

for taskFile in $taskFiles; do

set=`basename $taskFile`


echo $mcwsjav_mlf

# MLF transcription correction
# taken from HTK baseline script
sed -e '
# dos to unix line feed conversion
s/\x0D$//' \
-e "
            s/\x60//g              # remove unicode character grave accent.
       " \
-e "
            # fix the single quote for the word yield
            # and the quoted ROOTS
            # e.g. yield' --> yield
            # reason: YIELD' is not in dict, while YIELD is
            s/YIELD'/YIELD/g
            s/'ROOTS'/ROOTS/g
            s/'WHERE/WHERE/g
            s/PEOPLE'/PEOPLE/g
            s/SIT'/SIT/g
            s/'DOMINEE/DOMINEE/g
            s/CHURCH'/CHURCH/g" \
-e '
              # fix the single missing double full stop issue at the end of an utterance
              # e.g. I. C. N should be  I. C. N.
              # reason: N is not in dict, while N. is
              /^[A-Z]$/ {
              # append a line
                      N
              # search for single dot on the second line
                      /\n\./ {
              # found it - now replace the
                              s/\([A-Z]\)\n\./\1\.\n\./
                      }
              }' \
$mcwsjav_mlf |\
perl $local/mlf2text.pl > $dir2/$set.txt1

#exit

#taskFile=$taskFileDir/$set
# contains pointer to wav files with relative path --> add absolute path
echo taskFile = $taskFile
awk '{print "'$RWSJ'"$1}' < $taskFile > $dir2/${set}.flist || exit 1;

# this is like flist2scp.pl but it can take wav file list as input
(perl -e 'while(<>){
    m:^\S+/[\w\-]*_(T\w{6,7})\.wav$: || die "Bad line $_";
    $id = lc $1;
    print "$id $_";
}' < $dir2/$set.flist || exit 1) | sort > $dir2/${set}_wav.scp


# Make the utt2spk and spk2utt files.
cat $dir2/${set}_wav.scp | awk '{print $1, $1}' > $dir2/$set.utt2spk || exit 1;
cat $dir2/$set.utt2spk | $utils/utt2spk_to_spk2utt.pl > $dir2/$set.spk2utt || exit 1;

awk '{print $1}' < $dir2/$set.utt2spk |\
$local/find_transcripts_txt.pl $dir2/$set.txt1 | sort | uniq > $dir2/$set.txt
#rm $dir2/$set.txt1

# Create directory structure required by decoding scripts

cd $root
mkdir -p data/$dataset/$set
cp $dir2/${set}_wav.scp data/$dataset/$set/wav.scp || exit 1;
cp $dir2/$set.txt data/$dataset/$set/text || exit 1;
cp $dir2/$set.spk2utt data/$dataset/$set/spk2utt || exit 1;
cp $dir2/$set.utt2spk data/$dataset/$set/utt2spk || exit 1;

echo "Data preparation for $set succeeded"
#echo "Put files into $dir2/$set.*"


mfccdir=mfcc/$dataset
#for x in test_eval92_clean test_eval92_5k_clean dev_dt_05_clean dev_dt_20_clean train_si84_clean; do
#for x in si_tr; do
steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 \
  data/$dataset/$set exp/make_mfcc/$dataset/$set $mfccdir || exit 1;
steps/compute_cmvn_stats.sh data/$dataset/$set exp/make_mfcc/$dataset/$set $mfccdir || exit 1;

done
