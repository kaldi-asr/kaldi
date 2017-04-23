#!/usr/bin/env bash



if [ $# != 2 ]; then
  echo "Usage: $0 <data-dir> <num-to-split>"
  echo "E.g.: $0 data/train_cifar10 5"
  echo "This script creates split-up versions of images.scp and labels.txt in"
  echo "(in the example) data/train_cifar10/split5/{1,2,3,4,5}/{images.scp,labels.txt}."
  exit 1
fi

data=$1
numsplit=$2


if ! [ "$numsplit" -gt 0 ]; then
  echo "Invalid num-split argument $numsplit";
  exit 1;
fi

n=0;


s1=$data/split${numsplit}/1
if [ ! -d $s1 ]; then
  need_to_split=true
else
  need_to_split=false
  for f in images.scp labels.txt; do
    if [[ -f $data/$f && ( ! -f $s1/$f || $s1/$f -ot $data/$f ) ]]; then
      need_to_split=true
    fi
  done
fi

if ! $need_to_split; then
  exit 0;
fi

labels=$(for n in `seq $numsplit`; do echo $data/split${numsplit}${utt}/$n/labels.txt; done)

images=$(for n in `seq $numsplit`; do echo $data/split${numsplit}${utt}/$n/images.scp; done)

directories=$(for n in `seq $numsplit`; do echo $data/split${numsplit}${utt}/$n; done)

# if this mkdir fails due to argument-list being too long, iterate.
if ! mkdir -p $directories >&/dev/null; then
  for n in `seq $numsplit`; do
    mkdir -p $data/split${numsplit}${utt}/$n
  done
fi

# If lockfile is not installed, just don't lock it.  It's not a big deal.
which lockfile >&/dev/null && lockfile -l 60 $data/.split_lock
trap 'rm -f $data/.split_lock' EXIT HUP INT PIPE TERM

utils/split_scp.pl $data/labels.txt $labels || exit 1

utils/split_scp.pl $data/images.scp $images || exit 1


exit 0

