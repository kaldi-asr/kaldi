#!/usr/bin/env bash

data=$1
numsplit=$2

s1=$data/split${numsplit}/1
if [ ! -d $s1 ]; then
  need_to_split=true
else
  need_to_split=false
  for f in images.scp text utt2spk data/images.ark; do
    if [[ -f $data/$f && ( ! -f $s1/$f || $s1/$f -ot $data/$f ) ]]; then
      need_to_split=true
    fi
  done
fi

if ! $need_to_split; then
  exit 0;
fi

images=$(for n in `seq $numsplit`; do echo $data/split${numsplit}${utt}/$n/images.scp; done)
text=$(for n in `seq $numsplit`; do echo $data/split${numsplit}${utt}/$n/text; done)
utt2spk=$(for n in `seq $numsplit`; do echo $data/split${numsplit}${utt}/$n/utt2spk; done)
directories=$(for n in `seq $numsplit`; do echo $data/split${numsplit}${utt}/$n/data; done)

# if this mkdir fails due to argument-list being too long, iterate.
if ! mkdir -p $directories >&/dev/null; then
  for n in `seq $numsplit`; do
    #mkdir -p $data/split${numsplit}${utt}/$n
    mkdir -p $data/split${numsplit}${utt}/$n/data
  done
fi

# If lockfile is not installed, just don't lock it.  It's not a big deal.
which lockfile >&/dev/null && lockfile -l 60 $data/.split_lock
trap 'rm -f $data/.split_lock' EXIT HUP INT PIPE TERM

utils/split_scp.pl $data/images.scp $images || exit 1
utils/split_scp.pl $data/text $text || exit 1
utils/split_scp.pl $data/utt2spk $utt2spk || exit 1

exit 0

