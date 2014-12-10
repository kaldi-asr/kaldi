#!/bin/bash

# Copyright 2013-2014  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

# This script is as utils/copy_data_dir.sh in that it copies a data-dir,
# but it supports the --utts-per-spk-max option.  If nonzero, it modifies
# the utt2spk and spk2utt files by splitting each speaker into multiple
# versions, so that each speaker has no more than --utts-per-spk-max
# utterances.


# begin configuration section
utts_per_spk_max=-1
# end configuration section

. utils/parse_options.sh

if [ $# != 2 ]; then
  echo "Usage: "
  echo "  $0 [options] <srcdir> <destdir>"
  echo "e.g.:"
  echo " $0 --utts-per-spk-max 2 data/train data/train-max2"
  echo "Options"
  echo "   --utts-per-spk-max <n>  # number of utterances per speaker maximum,"
  echo "                           # default -1 (meaning no maximum).  E.g. 2."
  exit 1;
fi


export LC_ALL=C

srcdir=$1
destdir=$2

if [ ! -f $srcdir/utt2spk ]; then
  echo "$0: no such file $srcdir/utt2spk" 
  exit 1;
fi

set -e;
set -o pipefail

mkdir -p $destdir


if [ "$utts_per_spk_max" != -1 ]; then
  # create spk2utt file with reduced number of utterances per speaker.
  awk -v max=$utts_per_spk_max '{ n=2; count=0;
    while(n<=NF) {
      int_max=int(max)+ (rand() < (max-int(max))?1:0);
      nmax=n+int_max; count++; printf("%s-%06x", $1, count);
      for (;n<nmax&&n<=NF; n++) printf(" %s", $n); print "";} }' \
   <$srcdir/spk2utt >$destdir/spk2utt
  utils/spk2utt_to_utt2spk.pl <$destdir/spk2utt >$destdir/utt2spk

  if [ -f $srcdir/cmvn.scp ]; then
    # below, the first apply_map command outputs a cmvn.scp indexed by utt;
    # the second one outputs a cmvn.scp indexed by new speaker-id.
    utils/apply_map.pl -f 2 $srcdir/cmvn.scp <$srcdir/utt2spk | \
      utils/apply_map.pl -f 1 $destdir/utt2spk | sort | uniq > $destdir/cmvn.scp
    echo "$0: mapping cmvn.scp, but you may want to recompute it if it's needed,"
    echo " as it would probably change."
  fi
  if [ -f $srcdir/spk2gender ]; then
    utils/apply_map.pl -f 2 $srcdir/spk2gender <$srcdir/utt2spk | \
      utils/apply_map.pl -f 1 $destdir/utt2spk | sort | uniq >$destdir/spk2gender
  fi
else
  cp $srcdir/spk2utt $srcdir/utt2spk $destdir/
  [ -f $srcdir/spk2gender ] && cp $srcdir/spk2gender $destdir/
  [ -f $srcdir/cmvn.scp ] && cp $srcdir/cmvn.scp $destdir/
fi


for f in feats.scp segments wav.scp reco2file_and_channel text stm glm ctm; do
  [ -f $srcdir/$f ] && cp $srcdir/$f $destdir/
done

echo "$0: copied data from $srcdir to $destdir, with --utts-per-spk-max $utts_per_spk_max"
opts=
[ ! -f $srcdir/feats.scp ] && opts="--no-feats"
[ ! -f $srcdir/text ] && opts="$opts --no-text"

utils/validate_data_dir.sh $opts $destdir
