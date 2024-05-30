#!/usr/bin/env bash

# Copyright  2015 Tokyo Institute of Technology (Authors: Takafumi Moriya and Takahiro Shinozaki)
#            2015 Mitsubishi Electric Research Laboratories (Author: Shinji Watanabe)
# Apache 2.0
# Acknowledgement  This work was supported by JSPS KAKENHI Grant Number 26280055.

# Official evaluation data set (each set contains 10 speakers) preparation

# To be run from one directory above this script.

# The input is directory containing the official evaluation test set and transcripts.

if [ $# -ne 2 ]; then
  echo "Usage: "`basename $0`" <transcription-dir> <eval_num>"
  echo "See comments in the script for more details"
  exit 1
fi

tdir=$1
eval_num=$2
. ./path.sh

dir=data/local/$eval_num
mkdir -p $dir

cat $tdir/$eval_num/*/*-wav.list | sort > $dir/wav.flist
n=`cat $dir/wav.flist | wc -l`


sed -e 's?.*/??' -e 's?.wav??' -e 's?\-[R,L]??' $dir/wav.flist | paste - $dir/wav.flist \
  > $dir/wavflist.scp

awk '{
 printf("%s cat %s |\n", $1, $2);
}' < $dir/wavflist.scp | sort > $dir/wav.scp || exit 1;




# Get segments file...
# segments file format is: utt-id start-time end-time, e.g.:
# A01F0055_00380213_00385.951 => A01F0055 00380.213 00385.951

awk '{
      spkutt_id=$1;
      split(spkutt_id,T,"[_ ]");
      name=T[1]; stime=$2; etime=$3;
      printf("%s_%07.0f_%07.0f",name, int(1000*stime), int(1000*etime));
      for(i=4;i<=NF;i++) printf(" %s", tolower($i)); printf "\n"
}' $tdir/$eval_num/*/*-trans.text | sort > $dir/transcripts_${eval_num}.txt

# Remove option
cat $dir/transcripts_${eval_num}.txt \
 | perl -ane 's:\<s\>::gi;
               s:\<\/s\>::gi;
                print;' \
 | awk '{if(NF > 1) { print; } } ' > $dir/text

export LC_ALL=C;
sort -c $dir/text || exit 1; # check it's sorted.

## Create segment file
awk '{
       segment=$1;
       split(segment,S,"[_]");
       spkid=S[1]; startf=S[2]; endf=S[3];
       print segment " " spkid " " startf/1000 " " endf/1000
   }' < $dir/text > $dir/segments

# create an utt2spk file that assumes each conversation side is a separate speaker.
awk '{segment=$1; split(segment,S,"[_]"); spkid=S[1]; print $1 " " spkid}' $dir/segments > $dir/utt2spk || exit 1;
sort -k 2 $dir/utt2spk | utils/utt2spk_to_spk2utt.pl > $dir/spk2utt || exit 1;

dest=data/$eval_num
mkdir -p $dest
for x in wav.scp segments text utt2spk spk2utt; do
  cp $dir/$x $dest/$x
done

utils/fix_data_dir.sh $dest

if [ $(wc -l < $dest/wav.scp) -ne 10 ]; then
    echo "$0: error: expected 10 lines in wav.scp, got $(wc -l < $dest/wav.scp)"
  exit 1;
fi

echo "Completed preparation evaluation set $eval_num"
