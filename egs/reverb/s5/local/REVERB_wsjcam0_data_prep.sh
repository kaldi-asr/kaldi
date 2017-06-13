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

dir=$PWD/data/local/data
lmdir=$PWD/data/local/nist_lm
mkdir -p $dir $lmdir
local=$PWD/local
utils=$PWD/utils
root=$PWD

. ./path.sh # Needed for KALDI_ROOT
export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
   exit 1;
fi

RWSJ=$1 # input corpus (original or processed, tr or dt, etc.)
dataset=REVERB_dt # the name of the dataset to be created
if [ ! -z "$2" ]; then
   dataset=$2
fi
dt_or_x=dt # the WSJCAM0 set that the set is based on (tr, dt, ...)
# this will be used to find the correct transcriptions etc.
if [ ! -z "$3" ]; then
   dt_or_x=$3
fi

if [ ! -d "$RWSJ" ]; then
    echo Could not find directory $RWSJ! Check pathnames in corpus.sh!
    exit 1
fi

cd $dir
MIC=primary

# unfortunately, we need a pointer to HTK baseline
# since the corpus does NOT contain the data set descriptions
# for the REVERB Challenge
taskFileDir=$dir/../reverb_tools/ReleasePackage/reverb_tools_for_asr_ver2.0/taskFiles/1ch
#taskFiles=`ls $taskFileDir/*Data_dt_for_*`
nch=1
if [ "$dt_or_x" = "tr" ]; then
    taskFiles=`ls $taskFileDir/SimData_tr_for_${nch}ch*` || exit 1
else
    taskFiles=`ls $taskFileDir/SimData_${dt_or_x}_for_${nch}ch_{far,near}*` || exit 1
fi
for taskFile in $taskFiles; do

set=`basename $taskFile`

#taskFile=$taskFileDir/$set
dir2=$dir/$dataset
mkdir -p $dir2
# contains pointer to wav files with relative path --> add absolute path
echo taskFile = $taskFile
awk '{print "'$RWSJ/data'"$1}' < $taskFile > $dir2/${set}.flist || exit 1;

# this is like flist2scp.pl but it can take wav file list as input
perl -e 'while(<>){
    m:^\S+/(\w{8})\w*\.wav$: || die "Bad line $_";
    $id = lc $1;
    print "$id $_";
}' < $dir2/$set.flist | sort > $dir2/${set}_wav.scp || exit 1;

# find transcriptions of given utterances in si_dt.dot
# create a trans1 file for each set, convert to txt (kaldi "MLF")
dot=$dir/si_${dt_or_x}.dot
perl -e 'while (<>) { chomp; if (m/\/(\w{8})[^\/]+$/) { print $1, "\n"; } }' $taskFile |\
perl $local/find_transcripts_singledot.pl $dot \
> $dir2/$set.trans1 || exit 1;

noiseword="<NOISE>";
cat $dir2/$set.trans1 | $local/normalize_transcript.pl $noiseword | sort | uniq > $dir2/$set.txt || exit 1;
#exit


# Make the utt2spk and spk2utt files.
cat $dir2/${set}_wav.scp | awk '{print $1, $1}' > $dir2/$set.utt2spk || exit 1;
cat $dir2/$set.utt2spk | $utils/utt2spk_to_spk2utt.pl > $dir2/$set.spk2utt || exit 1;

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
