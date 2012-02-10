#!/bin/bash

# This script makes sure that only the segments present in 
# all of "feats.scp", "wav.scp" [if present], segments[if prsent]
# text, and utt2spk are present in any of them.
# It puts the original contents of data-dir into 
# data-dir/.backup

if [ $# != 1 ]; then
  echo "Usage: fix_data_dir.sh data-dir"
  exit 1
fi

data=$1
mkdir -p $data/.backup

cat $data/utt2spk | awk '{print $1}' > $data/utts
for x in feats.scp wav.scp text segments; do
  if [ -f $data/$x ]; then
     scripts/filter_scp.pl $data/$x $data/utts > $data/utts.tmp
     mv $data/utts.tmp $data/utts
  fi
done
[ ! -s $data/utts ] && echo "fix_data_dir.sh: no utterances remained: not doing anything." && \
   rm $data/utts && exit 1;

nutts=`cat $data/utts | wc -l`
nfeats=`cat $data/feats.scp | wc -l`
ntext=`cat $data/text | wc -l`
if [ "$nutts" -ne "$nfeats" -o "$nutts" -ne "$ntext" ]; then
  echo "fix_data_dir.sh: kept $nutts utterances, vs. $nfeats features and $ntext transcriptions."
else
  echo "fix_data_dir.sh: kept all $nutts utterances."
fi

for x in utt2spk feats.scp wav.scp text segments; do
  if [ -f $data/$x ]; then
     mv $data/$x $data/.backup/$x
     scripts/filter_scp.pl $data/utts $data/.backup/$x > $data/$x
  fi
done
scripts/utt2spk_to_spk2utt.pl $data/utt2spk > $data/spk2utt

rm $data/utts

echo "fix_data_dir.sh: old files are kept in $data/.backup"
