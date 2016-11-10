#!/bin/bash

# Copyright 2013  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0


# This script allows you to specify a 'segments' file with segments
# relative to existing utterances, with lines like
#  utterance_foo-1 utterance_foo 7.5 8.2
#  utterance_foo-2 utterance_foo 8.9 10.1
# and a 'text' file with sub-segmented text like
#  utterance_foo-1 hello there
#  utterance_foo-2 how are you
# and combine this with an existing data-dir that was all relative
# to the original utterance-ids like 'utterance_foo', producing
# a new subsegmented output directory.
#
# It does the right thing for you on the various files that the
# data directory contained (except you have to recreate
# the CMVN stats).


segment_end_padding=0.0

. utils/parse_options.sh

if [ $# != 4 ]; then
  echo "Usage: "
  echo "  $0 [options] <srcdir> <subsegments-file> <text-file> <destdir>"
  echo "This script sub-segments a data directory.  <subsegments-file> is to"
  echo "have lines of the form <new-utt> <old-utt> <start-time-within-old-utt> <end-time-within-old-utt>"
  echo "and <text-file> is of the form <new-utt> <word1> <word2> ... <wordN>."
  echo "This script appropriately combines the <subsegments-file> with the original"
  echo "segments file, if necessary, and if not, creates a segments file."
  echo "e.g.:"
  echo " $0 data/train [options] exp/tri3b_resegment/segments exp/tri3b_resegment/text data/train_resegmented"
  echo " Options:"
  echo "  --segment-end-padding <padding-time>       # e.g. 0.02.  Default 0.0.  If provided,"
  echo "                                             # we will add this value to the end times of <destdir>/segments"
  echo "                                             # when creating it.  This can be useful to account for"
  echo "                                             # end effects in feature generation.  The reason this is"
  echo "                                             # not just applied to the input segments file, is that"
  echo "                                             # for purposes of computing the num-frames of the parts of"
  echo "                                             # matrices in feats.scp, the padding should not be done."
  exit 1;
fi


export LC_ALL=C

srcdir=$1
subsegments=$2
new_text=$3
dir=$4


for f in "$subsegments" "$new_text" "$srcdir/utt2spk"; do
  if [ ! -f "$f" ]; then
    echo "$0: no such file $f"
    exit 1;
  fi
done

if ! mkdir -p $dir; then
  echo "$0: failed to create directory $dir"
fi

if ! cmp <(awk '{print $1}' <$subsegments)  <(awk '{print $1}' <$new_text); then
  echo "$0: expected the first fields of the files $subsegments and $new_text to be identical"
  exit 1
fi

# create the utt2spk in $dir
if ! awk '{if (NF != 4 || !($4 > $3)) { print("Bad line: " $0); exit(1) } }' <$subsegments; then
  echo "$0: failed checking subsegments file $subsegments"
  exit 1
fi

set -e
set -o pipefail

# Create a mapping from the new to old utterances.  This file will be deleted later.
awk '{print $1, $2}' < $subsegments > $dir/new2old_utt

# Create the new utt2spk file [just map from the second field
utils/apply_map.pl -f 2 $srcdir/utt2spk < $dir/new2old_utt >$dir/utt2spk
# .. and the new spk2utt file.
utils/utt2spk_to_spk2utt.pl  <$dir/utt2spk >$dir/spk2utt
# the new text file is just what the user provides.
cp $new_text $dir/text

# copy the source wav.scp
cp $srcdir/wav.scp $dir
if [ -f $srcdir/reco2file_and_channel ]; then
  cp $srcdir/reco2file_and_channel $dir
fi

if [ -f $srcdir/segments ]; then
  # we have to map the segments file.
  # What's going on below is a little subtle.
  # $srcdir/segments has lines like: <old-utt-id> <recording-id> <start-time> <end-time>
  # and $subsegments has lines like: <new-utt-id> <old-utt-id> <start-time> <end-time>
  # The apply-map command replaces <old-utt-id> [the 2nd field of $subsegments]
  # with <recording-id> <start-time> <end-time>.
  # so after that first command we have lines like
  # <new-utt-id> <recording-id> <start-time-of-old-utt-within-recording> <end-time-old-utt-within-recording> \
  #   <start-time-of-new-utt-within-old-utt> <end-time-of-new-utt-within-old-utt>
  # which the awk command turns into:
  # <new-utt-id> <recording-id> <start-time-of-new-utt-within-recording> <end-time-of-new-utt-within-recording>
  utils/apply_map.pl -f 2 $srcdir/segments <$subsegments | \
    awk -v pad=$segment_end_padding '{ print $1, $2, $5+$3, $6+$3+pad; }' >$dir/segments
else
  # the subsegments file just becomes the segments file.
  awk -v pad=$segment_end_padding '{$4 += pad; print}' <$subsegments >$dir/segments
fi

if [ -f $srcdir/utt2uniq ]; then
  utils/apply_map.pl -f 2 $srcdir/utt2uniq <$dir/new2old_utt >$dir/utt2uniq
fi

if [ -f $srcdir/feats.scp ]; then
  # We want to avoid recomputing the features.   We'll use sub-matrices of the
  # original feature matrices, using the [] notation that is available for
  # matrices in Kaldi.
  frame_shift=$(utils/data/get_frame_shift.sh $srcdir)
  echo "$0: note: frame shift is $frame_shift [affects feats.scp]"


  # The subsegments format is <new-utt-id> <old-utt-id> <start-time> <end-time>.
  # e.g. 'utt_foo-1 utt_foo 7.21 8.93'
  # The first awk command replaces this with the format:
  # <new-utt-id> <old-utt-id> <first-frame> <last-frame>
  # e.g. 'utt_foo-1 utt_foo 721 893'
  # and the apply_map.pl command replaces 'utt_foo' (the 2nd field) with its corresponding entry
  # from the original wav.scp, so we get a line like:
  # e.g. 'utt_foo-1 foo-bar.ark:514231 721 892'
  # Note: the reason we subtract one from the last time is that it's going to
  # represent the 'last' frame, not the 'end' frame [i.e. not one past the last],
  # in the matlab-like, but zero-indexed [first:last] notion.  For instance, a segment with 1 frame
  # would have start-time 0.00 and end-time 0.01, which would become the frame range
  # [0:0]
  # The second awk command turns this into something like
  # utt_foo-1 foo-bar.ark:514231[721:892]
  # It has to be a bit careful because the format actually allows for more general things
  # like pipes that might contain spaces, so it has to be able to produce output like the
  # following:
  # utt_foo-1 some command|[721:892]
  # Lastly, utils/data/normalize_data_range.pl will only do something nontrivial if
  # the original data-dir already had data-ranges in square brackets.
  awk -v s=$frame_shift '{print $1, $2, int(($3/s)+0.5), int(($4/s)-0.5);}' <$subsegments| \
    utils/apply_map.pl -f 2 $srcdir/feats.scp | \
    awk '{p=NF-1; for (n=1;n<NF-2;n++) printf("%s ", $n); k=NF-2; l=NF-1; printf("%s[%d:%d]\n", $k, $l, $NF)}' | \
    utils/data/normalize_data_range.pl  >$dir/feats.scp
fi


if [ -f $dir/cmvn.scp ]; then
  rm $dir/cmvn.scp
  echo "$0: warning: removing $dir/cmvn.scp, you will have to regenerate it from the features."
fi

# remove the utt2dur file in case it's now invalid-- it be regenerated from the segments file.
rm $dir/utt2dur 2>/dev/null || true

if [ -f $srcdir/spk2gender ]; then
  cp $srcdir/spk2gender $dir
fi
if [ -f $srcdir/glm ]; then
  cp $srcdir/glm $dir
fi

for f in stm ctm; do
  if [ -f $srcdir/$f ]; then
    echo "$0: not copying $srcdir/$f to $dir because sub-segmenting it is "
    echo " ... not implemented yet (and probably it's not needed.)"
  fi
done

rm $dir/new2old_utt

echo "$0: calling fix_data_dir.sh to remove any unused speakers, utterances, etc."
utils/data/fix_data_dir.sh $dir

validate_opts=
[ ! -f $srcdir/feats.scp ] && validate_opts="$validate_opts --no-feats"
[ ! -f $srcdir/wav.scp ] && validate_opts="$validate_opts --no-wav"

utils/data/validate_data_dir.sh $validate_opts $dir

echo "$0: subsegmented data from $srcdir to $dir"

