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

[ ! -d $data ] && echo "$0: no such directory $data" && exit 1;

[ ! -f $data/utt2spk ] && echo "$0: no such file $data/utt2spk" && exit 1;

cat $data/utt2spk | awk '{print $1}' > $data/utts

# Do a check.
export LC_ALL=C
! cat $data/utt2spk | sort | cmp - $data/utt2spk && \
   echo "utt2spk is not in sorted order (fix this yourelf)" && exit 1;

! cat $data/utt2spk | sort -k2 | cmp - $data/utt2spk && \
   echo "utt2spk is not in sorted order when sorted first on speaker-id " && \
   echo "(fix this by making speaker-ids prefixes of utt-ids)" && exit 1;

! cat $data/spk2utt | sort | cmp - $data/spk2utt && \
   echo "spk2utt is not in sorted order (fix this yourelf)" && exit 1;

maybe_wav=
[ ! -f $data/segments ] && maybe_wav=wav  # wav indexed by utts only if segments does not exist.
for x in feats.scp text segments $maybe_wav; do
  if [ -f $data/$x ]; then
     utils/filter_scp.pl $data/$x $data/utts > $data/utts.tmp
     mv $data/utts.tmp $data/utts
  fi
done
[ ! -s $data/utts ] && echo "fix_data_dir.sh: no utterances remained: not doing anything." && \
   rm $data/utts && exit 1;

nutts=`cat $data/utts | wc -l`
if [ -f $data/feats.scp ]; then
  nfeats=`cat $data/feats.scp | wc -l`
else
  nfeats=0
fi
ntext=`cat $data/text | wc -l`
if [ "$nutts" -ne "$nfeats" -o "$nutts" -ne "$ntext" ]; then
  echo "fix_data_dir.sh: kept $nutts utterances, vs. $nfeats features and $ntext transcriptions."
else
  echo "fix_data_dir.sh: kept all $nutts utterances."
fi

for x in utt2spk feats.scp text segments $maybe_wav; do
  if [ -f $data/$x ]; then
     mv $data/$x $data/.backup/$x
     utils/filter_scp.pl $data/utts $data/.backup/$x > $data/$x
  fi
done


if [ -f $data/segments ]; then
  awk '{print $2}' $data/segments | sort | uniq > $data/reco # reco means the id's of the recordings.
  [ -f $data/wav.scp ] && mv $data/wav.scp $data/.backup/ && \
    utils/filter_scp.pl $data/reco $data/.backup/wav.scp >$data/wav.scp
  [ -f $data/reco2file_and_channel ] && mv $data/reco2file_and_channel $data/.backup/ && \
    utils/filter_scp.pl $data/reco $data/.backup/reco2file_and_channel >$data/reco2file_and_channel
  rm $data/reco
fi

utils/utt2spk_to_spk2utt.pl $data/utt2spk > $data/spk2utt

rm $data/utts

echo "fix_data_dir.sh: old files are kept in $data/.backup"
