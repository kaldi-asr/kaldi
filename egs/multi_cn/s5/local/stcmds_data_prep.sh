#!/usr/bin/env bash

# Copyright 2019 Xingyu Na
# Apache 2.0

. ./path.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <corpus-path> <data-path>"
  echo " $0 /export/a05/xna/data/stcmds data/stcmds"
  exit 1;
fi

corpus=$1/ST-CMDS-20170001_1-OS
data=$2

if [ ! -d $corpus ]; then
  echo "Error: $0 requires complete corpus"
  exit 1;
fi

echo "**** Creating ST-CMDS data folder ****"

mkdir -p $data/{tmp,train}

# find wav audio file for train

find $corpus -iname "*.wav" > $data/tmp/wav.list
n=`cat $data/tmp/wav.list | wc -l`

[ $n -ne 102600 ] && \
  echo Warning: expected 102600 data files, found $n

split -l 10260 $data/tmp/wav.list $data/tmp/wav.list.x
for f in $data/tmp/wav.list.x*; do
  xname=$(basename ${f##*.})
  cat $f | awk -F'20170001' '{print $NF}' | awk -F'.' '{print $1}' > $data/tmp/utt.list.$xname
  cat $data/tmp/utt.list.$xname | awk '{print substr($1,1,6)}' > $data/tmp/spk.list.$xname
  paste -d' ' $data/tmp/utt.list.$xname $f > $data/tmp/wav.scp.$xname
  paste -d' ' $data/tmp/utt.list.$xname $data/tmp/spk.list.$xname > $data/tmp/utt2spk.$xname

  while read line; do
    tn=`dirname $line`/`basename $line .wav`.txt;
    cat $tn; echo;
  done < $data/tmp/wav.list.$xname > $data/tmp/text.list.$xname &
done
wait
for f in $data/tmp/text.list.*; do
  xname=$(basename ${f##*.})
  paste -d' ' $data/tmp/utt.list.$xname $f |\
    sed 's/，//g' |\
    python local/word_segment.py |\
    tr '[a-z]' '[A-Z]' |\
    awk '{if (NF > 1) print $0;}' > $data/tmp/text.split.$xname &
done
wait

cat $data/tmp/text.split.* | sort > $data/train/text
cat $data/tmp/utt2spk.* | sort > $data/train/utt2spk
cat $data/tmp/wav.scp.* | sort > $data/train/wav.scp
utils/utt2spk_to_spk2utt.pl $data/train/utt2spk > $data/train/spk2utt
utils/data/validate_data_dir.sh --no-feats $data/train

exit 0
