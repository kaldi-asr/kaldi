#!/bin/bash
# Copyright 2010-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0


# This script operates on a directory, such as in data/train/,
# that contains some subset of the following files:
#  feats.scp
#  wav.scp
#  spk2utt
#  utt2spk
#  text
# It creates a subset of that data, consisting of some specified
# number of utterances.  (The selected utterances are distributed
# evenly throughout the file, by the program ./subset_scp.pl).


# There are three options, none compatible with any other.

# If you give the --per-spk option, it will attempt to select the supplied
# number of utterances for each speaker (typically you would supply a much
# smaller number in this case).

# If you give the --shortest option, it will give you the n shortest utterances.

# If you give the --first option it will just give you the n first utterances.

shortest=false
perspk=false
first_opt=""

if [ "$1" == "--per-spk" ]; then
  perspk=true;
  shift;
elif [ "$1" == "--shortest" ]; then
  shortest=true;
  shift;
elif [ "$1" == "--first" ]; then
  first_opt="--first";
  shift;
elif [ "$1" == "--last" ]; then
  first_opt="--last";
  shift;
fi



if [ $# != 3 ]; then
  echo "Usage: subset_data_dir.sh [--per-spk] <srcdir> <num-utt> <destdir>"
  exit 1;
fi

srcdir=$1
numutt=$2
destdir=$3


if [ ! -f $srcdir/utt2spk ]; then
  echo "subset_data_dir.sh: no such file $srcdir/utt2spk" 
  exit 1;
fi


function do_filtering {
  # assumes the utt2spk and spk2utt files already exist.
  [ -f $srcdir/feats.scp ] && utils/filter_scp.pl $destdir/utt2spk <$srcdir/feats.scp >$destdir/feats.scp
  [ -f $srcdir/wav.scp ] && utils/filter_scp.pl $destdir/utt2spk <$srcdir/wav.scp >$destdir/wav.scp
  [ -f $srcdir/text ] && utils/filter_scp.pl $destdir/utt2spk <$srcdir/text >$destdir/text
  [ -f $srcdir/spk2gender ] && utils/filter_scp.pl $destdir/spk2utt <$srcdir/spk2gender >$destdir/spk2gender
  [ -f $srcdir/cmvn.scp ] && utils/filter_scp.pl $destdir/spk2utt <$srcdir/cmvn.scp >$destdir/cmvn.scp
  if [ -f $srcdir/segments ]; then
     utils/filter_scp.pl $destdir/utt2spk <$srcdir/segments >$destdir/segments
     awk '{print $2;}' $destdir/segments | sort | uniq > $destdir/reco # recordings.
     # The next line would override the command above for wav.scp, which would be incorrect.
     [ -f $srcdir/wav.scp ] && utils/filter_scp.pl $destdir/reco <$srcdir/wav.scp >$destdir/wav.scp
     [ -f $srcdir/reco2file_and_channel ] && \
       utils/filter_scp.pl $destdir/reco <$srcdir/reco2file_and_channel >$destdir/reco2file_and_channel
     rm $destdir/reco
  fi
  srcutts=`cat $srcdir/utt2spk | wc -l`
  destutts=`cat $destdir/utt2spk | wc -l`
  echo "Retained $numutt utterances per speaker from data-dir $srcdir and put it in $destdir, reducing #utt from $srcutts to $destutts"
}


## scripting note: $perspk evaluates to true or false
## so this becomes the command true or false.
if $perspk; then
  mkdir -p $destdir
  awk '{ n='$numutt'; printf("%s ",$1); skip=1; while(n*(skip+1) <= NF-1) { skip++; }
         for(x=2; x<=NF && x <= n*skip; x += skip) { printf("%s ", $x); } 
         printf("\n"); }' <$srcdir/spk2utt >$destdir/spk2utt
  utils/spk2utt_to_utt2spk.pl < $destdir/spk2utt > $destdir/utt2spk
  do_filtering; # bash function.
  exit 0;
else
  if [ $numutt -gt `cat $srcdir/feats.scp | wc -l` ]; then
    echo "subset_data_dir.sh: cannot subset to more utterances than you originally had."
    exit 1;
  fi 
  mkdir -p $destdir || exit 1;

  ## scripting note: $shortest evaluates to true or false
  ## so this becomes the command true or false.
  if $shortest; then
    # select the n shortest utterances.
    . ./path.sh
    [ ! -f $srcdir/feats.scp ] && echo "$0: you selected --shortest but no feats.scp exist." && exit 1;
    feat-to-len scp:$srcdir/feats.scp ark,t:$destdir/tmp.len || exit 1;
    sort -n -k2 $destdir/tmp.len | awk '{print $1}' | head -$numutt >$destdir/tmp.uttlist
    utils/filter_scp.pl $destdir/tmp.uttlist $srcdir/utt2spk >$destdir/utt2spk
    rm $destdir/tmp.uttlist $destdir/tmp.len
  else
    utils/subset_scp.pl $first_opt $numutt $srcdir/utt2spk > $destdir/utt2spk || exit 1;
  fi
  utils/utt2spk_to_spk2utt.pl < $destdir/utt2spk > $destdir/spk2utt
  do_filtering;
  exit 0;
fi
