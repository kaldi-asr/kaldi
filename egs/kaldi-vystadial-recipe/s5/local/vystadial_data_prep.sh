#!/bin/bash
# Author:   Ondrej Platek, Copyright 2012, code is without any warranty!
# Created:  11:06:13 16/11/2012
# Modified: 11:06:13 16/11/2012
#
#
# Makes train/test splits
# local/voxforge_data_prep.sh --nspk_test ${nspk_test} ${SELECTED} || exit 1
# create files: (TYPE=train|test)
#   a) ${TYPE}_trans.txt: ID transcription capitalized! No interputction
#   b) ${TYPE}_wav.scp: ID path2ID.wav 
#   c) $TYPE.utt2spk: ID-recording ID-speaker
#   d) $TYPE.spk2utt
#   e) $TYPE.spk2gender  all speakers are male
# we have ID-recording = ID-speaker

renice 20 $$


every_n=1
[ -f path.sh ] && . ./path.sh # source the path.
. utils/parse_options.sh || exit 1;


msg="Usage: $0 [--every-n 30] <data-directory>";
if [ $# -gt 1 ] ; then
    echo "$msg"; exit 1;
fi
if [ $# -eq 0 ] ; then
    echo "$msg"; exit 1;
fi

DATA=$1

echo "=== Starting initial Vystadial data preparation ..."
echo "--- Making test/train data split from $DATA taking every $every_n recording..."

locdata=data/local
loctmp=$locdata/tmp
rm -rf $loctmp >/dev/null 2>&1
mkdir -p $locdata
mkdir -p $loctmp

i=0
for d in test train ; do
    ls $DATA/$d/ | sed -n /.*wav$/p |\
    while read wav ; do
        # echo "DEBUGGING wav: $wav"
        ((i++)) # bash specific
        if [ $i -ge $every_n ] ; then
            i=0
            pwav=$DATA/$d/$wav
            echo "$wav $pwav" >> ${loctmp}/${d}_wav.scp.unsorted
            echo "$wav $wav" >> ${loctmp}/${d}.utt2spk.unsorted
            echo "$wav $wav" >> ${loctmp}/${d}.spk2utt.unsorted
            # transcribtion of $wav
            trn=`cat $DATA/$d/$wav.trn`
            # echo "DEBUGGING trn: $trn"
            echo "$wav $trn" >> ${loctmp}/${d}_trans.txt.unsorted
            echo "$wav M" >> ${loctmp}/spk2gender.unsorted
        fi
    done # while read wav 

    # Sorting
    for unsorted in _wav.scp.unsorted _trans.txt.unsorted \
        .spk2utt.unsorted .utt2spk.unsorted _wav.scp.unsorted
    do
       u="${d}${unsorted}"
       s=`echo "$u" | sed -e s:.unsorted::`
       sort "${loctmp}/$u" -k1 > "${locdata}/$s"
    done # for unsorted

    #### copy to data dir ###
    mkdir -p data/$d
    cp $locdata/${d}_wav.scp data/$d/wav.scp || exit 1;
    cp $locdata/${d}_trans.txt data/$d/text || exit 1;
    cp $locdata/$d.spk2utt data/$d/spk2utt || exit 1;
    cp $locdata/$d.utt2spk data/$d/utt2spk || exit 1;
done # for in test train

# should be set..OK for 1:1 spk2utt, spk from test AND train
sort "${loctmp}/spk2gender.unsorted" -k1 > "${locdata}/spk2gender" 
utils/filter_scp.pl data/$d/spk2utt $locdata/spk2gender > data/$d/spk2gender || exit 1;
