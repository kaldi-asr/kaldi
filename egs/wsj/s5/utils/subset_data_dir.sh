#!/bin/bash
# Copyright 2010-2011  Microsoft Corporation
#           2012-2013  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0


# This script operates on a data directory, such as in data/train/.
# See http://kaldi-asr.org/doc/data_prep.html#data_prep_data
# for what these directories contain.

# This script creates a subset of that data, consisting of some specified
# number of utterances.  (The selected utterances are distributed evenly
# throughout the file, by the program ./subset_scp.pl).

# There are six options, none compatible with any other.

# If you give the --per-spk option, it will attempt to select the supplied
# number of utterances for each speaker (typically you would supply a much
# smaller number in this case).

# If you give the --speakers option, it selects a subset of n randomly
# selected speakers.

# If you give the --shortest option, it will give you the n shortest utterances.

# If you give the --first option, it will just give you the n first utterances.

# If you give the --last option, it will just give you the n last utterances.

# If you give the --spk-list or --utt-list option, it reads the
# speakers/utterances to keep from <speaker-list-file>/<utt-list-file>" (note,
# in this case there is no <num-utt> positional parameter; see usage message.)


shortest=false
perspk=false
first_opt=""
speakers=false
spk_list_specified=false
utt_list_specified=false

if [ "$1" == "--per-spk" ]; then
  perspk=true;
  shift;
elif [ "$1" == "--shortest" ]; then
  shortest=true;
  shift;
elif [ "$1" == "--first" ]; then
  first_opt="--first";
  shift;
elif [ "$1" == "--speakers" ]; then
  speakers=true
  shift;
elif [ "$1" == "--last" ]; then
  first_opt="--last";
  shift;
elif [ "$1" == "--spk-list" ]; then
  spk_list_specified=true
  shift;
elif [ "$1" == "--utt-list" ]; then
  utt_list_specified=true
  shift;
fi




if [ $# != 3 ]; then
  echo "Usage: "
  echo "  subset_data_dir.sh [--speakers|--shortest|--first|--last|--per-spk] <srcdir> <num-utt> <destdir>"
  echo "  subset_data_dir.sh [--spk-list <speaker-list-file>] <srcdir> <destdir>"
  echo "  subset_data_dir.sh [--utt-list <utterance-list-file>] <srcdir> <destdir>"
  echo "By default, randomly selects <num-utt> utterances from the data directory."
  echo "With --speakers, randomly selects enough speakers that we have <num-utt> utterances"
  echo "With --per-spk, selects <num-utt> utterances per speaker, if available."
  echo "With --first, selects the first <num-utt> utterances"
  echo "With --last, selects the last <num-utt> utterances"
  echo "With --shortest, selects the shortest <num-utt> utterances."
  echo "With --spk-list, reads the speakers to keep from <speaker-list-file>"
  exit 1;
fi

if $spk_list_specified; then
  spk_list=$1
  srcdir=$2
  destdir=$3
elif $utt_list_specified; then
  utt_list=$1
  srcdir=$2
  destdir=$3
else
  srcdir=$1
  numutt=$2
  destdir=$3
fi


export LC_ALL=C

if [ ! -f $srcdir/utt2spk ]; then
  echo "subset_data_dir.sh: no such file $srcdir/utt2spk"
  exit 1;
fi

function do_filtering {
  # assumes the utt2spk and spk2utt files already exist.
  [ -f $srcdir/feats.scp ] && utils/filter_scp.pl $destdir/utt2spk <$srcdir/feats.scp >$destdir/feats.scp
  [ -f $srcdir/vad.scp ] && utils/filter_scp.pl $destdir/utt2spk <$srcdir/vad.scp >$destdir/vad.scp
  [ -f $srcdir/utt2lang ] && utils/filter_scp.pl $destdir/utt2spk <$srcdir/utt2lang >$destdir/utt2lang
  [ -f $srcdir/utt2dur ] && utils/filter_scp.pl $destdir/utt2spk <$srcdir/utt2dur >$destdir/utt2dur
  [ -f $srcdir/utt2num_frames ] && utils/filter_scp.pl $destdir/utt2spk <$srcdir/utt2num_frames >$destdir/utt2num_frames
  [ -f $srcdir/utt2uniq ] && utils/filter_scp.pl $destdir/utt2spk <$srcdir/utt2uniq >$destdir/utt2uniq
  [ -f $srcdir/wav.scp ] && utils/filter_scp.pl $destdir/utt2spk <$srcdir/wav.scp >$destdir/wav.scp
  [ -f $srcdir/spk2warp ] && utils/filter_scp.pl $destdir/spk2utt <$srcdir/spk2warp >$destdir/spk2warp
  [ -f $srcdir/utt2warp ] && utils/filter_scp.pl $destdir/utt2spk <$srcdir/utt2warp >$destdir/utt2warp
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

     # Filter the STM file for proper sclite scoring (this will also remove the comments lines)
     [ -f $srcdir/stm ] && utils/filter_scp.pl $destdir/reco < $srcdir/stm > $destdir/stm

     rm $destdir/reco
  else
     awk '{print $1;}' $destdir/wav.scp | sort | uniq > $destdir/reco
     [ -f $srcdir/reco2file_and_channel ] && \
       utils/filter_scp.pl $destdir/reco <$srcdir/reco2file_and_channel >$destdir/reco2file_and_channel
     
     rm $destdir/reco
  fi
  srcutts=`cat $srcdir/utt2spk | wc -l`
  destutts=`cat $destdir/utt2spk | wc -l`
  echo "$0: reducing #utt from $srcutts to $destutts"
}


if $spk_list_specified; then
  mkdir -p $destdir
  utils/filter_scp.pl "$spk_list" $srcdir/spk2utt > $destdir/spk2utt || exit 1;
  utils/spk2utt_to_utt2spk.pl < $destdir/spk2utt > $destdir/utt2spk || exit 1;
  do_filtering; # bash function.
  exit 0;
elif $utt_list_specified; then
  mkdir -p $destdir
  utils/filter_scp.pl "$utt_list" $srcdir/utt2spk > $destdir/utt2spk || exit 1;
  utils/utt2spk_to_spk2utt.pl < $destdir/utt2spk > $destdir/spk2utt || exit 1;
  do_filtering; # bash function.
  exit 0;
elif $speakers; then
  mkdir -p $destdir
  utils/shuffle_list.pl < $srcdir/spk2utt | awk -v numutt=$numutt '{ if (tot < numutt){ print; } tot += (NF-1); }' | \
    sort > $destdir/spk2utt
  utils/spk2utt_to_utt2spk.pl < $destdir/spk2utt > $destdir/utt2spk
  do_filtering; # bash function.
  exit 0;
elif $perspk; then
  mkdir -p $destdir
  awk '{ n='$numutt'; printf("%s ",$1); skip=1; while(n*(skip+1) <= NF-1) { skip++; }
         for(x=2; x<=NF && x <= n*skip; x += skip) { printf("%s ", $x); }
         printf("\n"); }' <$srcdir/spk2utt >$destdir/spk2utt
  utils/spk2utt_to_utt2spk.pl < $destdir/spk2utt > $destdir/utt2spk
  do_filtering; # bash function.
  exit 0;
else
  if [ $numutt -gt `cat $srcdir/utt2spk | wc -l` ]; then
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
