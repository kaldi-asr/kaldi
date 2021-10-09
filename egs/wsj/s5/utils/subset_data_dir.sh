#!/usr/bin/env bash
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
speakers=false
first_opt=
spk_list=
utt_list=

expect_args=3
case $1 in
  --first|--last) first_opt=$1; shift ;;
  --per-spk)  perspk=true; shift ;;
  --shortest) shortest=true; shift ;;
  --speakers) speakers=true; shift ;;
  --spk-list) shift; spk_list=$1; shift; expect_args=2 ;;
  --utt-list) shift; utt_list=$1; shift; expect_args=2 ;;
  --*) echo "$0: invalid option '$1'"; exit 1
esac

if [ $# != $expect_args ]; then
  echo "Usage:"
  echo "  subset_data_dir.sh [--speakers|--shortest|--first|--last|--per-spk] <srcdir> <num-utt> <destdir>"
  echo "  subset_data_dir.sh [--spk-list <speaker-list-file>] <srcdir> <destdir>"
  echo "  subset_data_dir.sh [--utt-list <utt-list-file>] <srcdir> <destdir>"
  echo "By default, randomly selects <num-utt> utterances from the data directory."
  echo "With --speakers, randomly selects enough speakers that we have <num-utt> utterances"
  echo "With --per-spk, selects <num-utt> utterances per speaker, if available."
  echo "With --first, selects the first <num-utt> utterances"
  echo "With --last, selects the last <num-utt> utterances"
  echo "With --shortest, selects the shortest <num-utt> utterances."
  echo "With --spk-list, reads the speakers to keep from <speaker-list-file>"
  echo "With --utt-list, reads the utterances to keep from <utt-list-file>"
  exit 1;
fi

srcdir=$1
if [[ $spk_list || $utt_list ]]; then
  numutt=
  destdir=$2
else
  numutt=$2
  destdir=$3
fi

export LC_ALL=C

if [ ! -f $srcdir/utt2spk ]; then
  echo "$0: no such file $srcdir/utt2spk"
  exit 1
fi

if [[ $numutt && $numutt -gt $(wc -l <$srcdir/utt2spk) ]]; then
  echo "$0: cannot subset to more utterances than you originally had."
  exit 1
fi

if $shortest && [ ! -f $srcdir/feats.scp ]; then
  echo "$0: you selected --shortest but no feats.scp exist."
  exit 1
fi

mkdir -p $destdir || exit 1

if [[ $spk_list ]]; then
  utils/filter_scp.pl "$spk_list" $srcdir/spk2utt > $destdir/spk2utt || exit 1;
  utils/spk2utt_to_utt2spk.pl < $destdir/spk2utt > $destdir/utt2spk || exit 1;
elif [[ $utt_list ]]; then
  utils/filter_scp.pl "$utt_list" $srcdir/utt2spk > $destdir/utt2spk || exit 1;
  utils/utt2spk_to_spk2utt.pl < $destdir/utt2spk > $destdir/spk2utt || exit 1;
elif $speakers; then
  utils/shuffle_list.pl < $srcdir/spk2utt |
    awk -v numutt=$numutt '{ if (tot < numutt){ print; } tot += (NF-1); }' |
    sort > $destdir/spk2utt
  utils/spk2utt_to_utt2spk.pl < $destdir/spk2utt > $destdir/utt2spk
elif $perspk; then
  awk '{ n='$numutt'; printf("%s ",$1);
         skip=1; while(n*(skip+1) <= NF-1) { skip++; }
         for(x=2; x<=NF && x <= (n*skip+1); x += skip) { printf("%s ", $x); }
         printf("\n"); }' <$srcdir/spk2utt >$destdir/spk2utt
  utils/spk2utt_to_utt2spk.pl < $destdir/spk2utt > $destdir/utt2spk
else
  if $shortest; then
    # Select $numutt shortest utterances.
    . ./path.sh
    if [ -f $srcdir/utt2num_frames ]; then
      ln -sf $(utils/make_absolute.sh $srcdir)/utt2num_frames $destdir/tmp.len
    else
      feat-to-len scp:$srcdir/feats.scp ark,t:$destdir/tmp.len || exit 1;
    fi
    sort -n -k2 $destdir/tmp.len |
      awk '{print $1}' |
      head -$numutt >$destdir/tmp.uttlist
    utils/filter_scp.pl $destdir/tmp.uttlist $srcdir/utt2spk >$destdir/utt2spk
    rm $destdir/tmp.uttlist $destdir/tmp.len
  else
    # Select $numutt random utterances.
    utils/subset_scp.pl $first_opt $numutt $srcdir/utt2spk > $destdir/utt2spk || exit 1;
  fi
  utils/utt2spk_to_spk2utt.pl < $destdir/utt2spk > $destdir/spk2utt
fi

# Perform filtering. utt2spk and spk2utt files already exist by this point.
# Filter by utterance.
[ -f $srcdir/feats.scp ] &&
  utils/filter_scp.pl $destdir/utt2spk <$srcdir/feats.scp >$destdir/feats.scp
[ -f $srcdir/vad.scp ] &&
  utils/filter_scp.pl $destdir/utt2spk <$srcdir/vad.scp >$destdir/vad.scp
[ -f $srcdir/utt2lang ] &&
  utils/filter_scp.pl $destdir/utt2spk <$srcdir/utt2lang >$destdir/utt2lang
[ -f $srcdir/utt2dur ] &&
  utils/filter_scp.pl $destdir/utt2spk <$srcdir/utt2dur >$destdir/utt2dur
[ -f $srcdir/utt2num_frames ] &&
  utils/filter_scp.pl $destdir/utt2spk <$srcdir/utt2num_frames >$destdir/utt2num_frames
[ -f $srcdir/utt2uniq ] &&
  utils/filter_scp.pl $destdir/utt2spk <$srcdir/utt2uniq >$destdir/utt2uniq
[ -f $srcdir/wav.scp ] &&
  utils/filter_scp.pl $destdir/utt2spk <$srcdir/wav.scp >$destdir/wav.scp
[ -f $srcdir/utt2warp ] &&
  utils/filter_scp.pl $destdir/utt2spk <$srcdir/utt2warp >$destdir/utt2warp
[ -f $srcdir/text ] &&
  utils/filter_scp.pl $destdir/utt2spk <$srcdir/text >$destdir/text

# Filter by speaker.
[ -f $srcdir/spk2warp ] &&
  utils/filter_scp.pl $destdir/spk2utt <$srcdir/spk2warp >$destdir/spk2warp
[ -f $srcdir/spk2gender ] &&
  utils/filter_scp.pl $destdir/spk2utt <$srcdir/spk2gender >$destdir/spk2gender
[ -f $srcdir/cmvn.scp ] &&
  utils/filter_scp.pl $destdir/spk2utt <$srcdir/cmvn.scp >$destdir/cmvn.scp

# Filter by recording-id.
if [ -f $srcdir/segments ]; then
  utils/filter_scp.pl $destdir/utt2spk <$srcdir/segments >$destdir/segments
  # Recording-ids are in segments.
  awk '{print $2}' $destdir/segments | sort | uniq >$destdir/reco
  # The next line overrides the command above for wav.scp, which would be incorrect.
  [ -f $srcdir/wav.scp ] &&
    utils/filter_scp.pl $destdir/reco <$srcdir/wav.scp >$destdir/wav.scp
else
  # No segments; recording-ids are in wav.scp.
  awk '{print $1}' $destdir/wav.scp | sort | uniq >$destdir/reco
fi

[ -f $srcdir/reco2file_and_channel ] &&
  utils/filter_scp.pl $destdir/reco <$srcdir/reco2file_and_channel >$destdir/reco2file_and_channel
[ -f $srcdir/reco2dur ] &&
  utils/filter_scp.pl $destdir/reco <$srcdir/reco2dur >$destdir/reco2dur

# Filter the STM file for proper sclite scoring.
# Copy over the comments from STM file.
[ -f $srcdir/stm ] &&
  (grep "^;;" $srcdir/stm
   utils/filter_scp.pl $destdir/reco $srcdir/stm) >$destdir/stm

rm $destdir/reco

# Copy frame_shift if present.
[ -f $srcdir/frame_shift ] && cp $srcdir/frame_shift $destdir

srcutts=$(wc -l <$srcdir/utt2spk)
destutts=$(wc -l <$destdir/utt2spk)
echo "$0: reducing #utt from $srcutts to $destutts"
exit 0
