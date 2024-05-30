#!/usr/bin/env bash

# Copyright 2013-2016  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

# This script copies a data directory (like utils/copy_data.sh) while
# modifying (splitting or merging) the speaker information in that data directory.
#
# This is done without looking at the data at all; we use only duration
# constraints and maximum-num-utts-per-speaker to assign contiguous
# sets of utterances to speakers.
#
# This has two general uses:
# (1) when dumping iVectors for training purposes, it's helpful to have
#   a good variety of iVectors, and this can be accomplished by splitting
#   speakers up into multiple copies of those speakers.  We typically
#   use the --utts-per-spk-max 2 option for this.
# (2) when dealing with data that is not diarized, and given that we
#   haven't checked any diarization scripts into Kaldi yet, this
#   script can do a "dumb" diarization that just groups consecutive
#   utterances into groups based on length constraints.
#   There are two cases here:

#       a) With --respect-speaker-info true (the default),
#         it only splits within existing speakers.
#         This is suitable when you have existing speaker
#         info that's meaningful in some way, e.g. represents
#         individual recordings.
#      b) With --respect-speaker-info false,
#        it completely ignores the existing speaker information
#        and constructs new speaker identities based on
#        utterance names.  This is suitable in scenarios when
#        you have a one-to-one map between speakers and
#        utterances.

# begin configuration section
utts_per_spk_max=-1
seconds_per_spk_max=-1
respect_speaker_info=true
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
  echo "   --seconds-per-spk-max <n> # number of seconds per speaker maximum,"
  echo "                             # default -1 (meaning no maximum).  E.g. 60."
  echo "   --respect-speaker-info <true|false>  # If true, respect the"
  echo "                                        # existing speaker map (i.e. do not"
  echo "                                        # assign utterances from different"
  echo "                                        # speakers to the same generated speaker)."
  echo "                                        # Default: true."
  echo "Note: one or both of the --utts-per-spk-max or --seconds-per-spk-max"
  echo "options is required."
  exit 1;
fi

export LC_ALL=C

srcdir=$1
destdir=$2

if [ "$destdir"  == "$srcdir" ]; then
  echo "$0: <srcdir> must be different from <destdir>."
  exit 1
fi

if [ "$seconds_per_spk_max" == "-1" ] && ! [ "$utts_per_spk_max" -gt 0 ]; then
  echo "$0: one or both of the --utts-per-spk-max or --seconds-per-spk-max options must be provided."
fi

if [ ! -f $srcdir/utt2spk ]; then
  echo "$0: no such file $srcdir/utt2spk"
  exit 1;
fi

set -e;
set -o pipefail

mkdir -p $destdir

if [ "$seconds_per_spk_max" != -1 ]; then
  # we need the utt2dur file.
  utils/data/get_utt2dur.sh $srcdir
  utt2dur_opt="--utt2dur=$srcdir/utt2dur"
else
  utt2dur_opt=
fi

utils/data/internal/modify_speaker_info.py \
   $utt2dur_opt --respect-speaker-info=$respect_speaker_info \
  --utts-per-spk-max=$utts_per_spk_max --seconds-per-spk-max=$seconds_per_spk_max \
  <$srcdir/utt2spk >$destdir/utt2spk

utils/utt2spk_to_spk2utt.pl <$destdir/utt2spk >$destdir/spk2utt

# This script won't create the new cmvn.scp, it should be recomputed.
if [ -f $destdir/cmvn.scp ]; then
  mkdir -p $destdir/.backup
  mv $destdir/cmvn.scp $destdir/.backup
  echo "$0: moving $destdir/cmvn.scp to $destdir/.backup/cmvn.scp"
fi

# these things won't be affected by the change of speaker mapping.
for f in feats.scp segments wav.scp reco2file_and_channel text stm glm ctm; do
  [ -f $srcdir/$f ] && cp $srcdir/$f $destdir/
done


orig_num_spk=$(wc -l <$srcdir/spk2utt)
new_num_spk=$(wc -l <$destdir/spk2utt)

echo "$0: copied data from $srcdir to $destdir, number of speakers changed from $orig_num_spk to $new_num_spk"
opts=
[ ! -f $srcdir/feats.scp ] && opts="--no-feats"
[ ! -f $srcdir/text ] && opts="$opts --no-text"
[ ! -f $srcdir/wav.scp ] && opts="$opts --no-wav"

utils/validate_data_dir.sh $opts $destdir
