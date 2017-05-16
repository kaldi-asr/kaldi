#!/bin/bash

# Copyright 2013  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

# This script copies and modifies a data directory while combining
# segments whose duration is lower than a specified minimum segment
# length.
#
# Note: this does not work for the wav.scp, since there is no natural way to
# concatenate segments; you have to operate on directories that already have
# features extracted.

#


# begin configuration section
cleanup=true
# end configuration section

. utils/parse_options.sh

if [ $# != 3 ]; then
  echo "Usage: "
  echo "  $0 [options] <srcdir> <min-segment-length-in-seconds> <dir>"
  echo "e.g.:"
  echo " $0 data/train 1.55 data/train_comb"
  # options documentation here.
  exit 1;
fi


export LC_ALL=C

srcdir=$1
min_seg_len=$2
dir=$3

if [ "$dir" == "$srcdir" ]; then
  echo "$0: this script requires <srcdir> and <dir> to be different."
  exit 1
fi

for f in $srcdir/utt2spk $srcdir/feats.scp; do
  [ ! -s $f ] && echo "$0: expected file $f to exist and be nonempty" && exit 1
done

if ! awk '{if (NF != 2) exit(1);}' <$srcdir/feats.scp; then
  echo "$0: could not combine short segments because $srcdir/feats.scp has "
  echo " entries with too many fields"
fi

if ! mkdir -p $dir; then
  echo "$0: could not create directory $dir"
  exit 1;
fi

if ! utils/validate_data_dir.sh $srcdir; then
  echo "$0: failed to validate input directory $srcdir.  If needed, run   utils/fix_data_dir.sh $srcdir"
  exit 1
fi

if ! python -c "x=float('$min_seg_len'); assert(x>0.0 and x<100.0);" 2>/dev/null; then
  echo "$0: bad <min-segment-length-in-seconds>: got '$min_seg_len'"
  exit 1
fi

set -e
set -o pipefail

# make sure $srcdir/utt2dur exists.
utils/data/get_utt2dur.sh $srcdir

utils/data/internal/choose_utts_to_combine.py --min-duration=$min_seg_len \
  $srcdir/spk2utt $srcdir/utt2dur $dir/utt2utts $dir/utt2spk $dir/utt2dur

utils/utt2spk_to_spk2utt.pl < $dir/utt2spk > $dir/spk2utt

# create the feats.scp.
# if a line of utt2utts is like 'utt2-comb2 utt2 utt3', then
# the utils/apply_map.pl will create a line that looks like
# 'utt2-comb2 foo.ark:4315 foo.ark:431423'
# and the awk command creates suitable command lines like:
# 'utt2-comb2 concat-feats foo.ark:4315 foo.ark:431423 - |'
utils/apply_map.pl -f 2- $srcdir/feats.scp <$dir/utt2utts | \
  awk '{if (NF<=2){print;} else { $1 = $1 " concat-feats --print-args=false"; $NF = $NF " - |"; print; }}' > $dir/feats.scp

# create $dir/text by concatenating the source 'text' entries for the original
# utts.
utils/apply_map.pl -f 2- $srcdir/text <$dir/utt2utts > $dir/text

if [ -f $srcdir/utt2uniq ]; then
  # the utt2uniq file is such that if 2 utts were derived from the same original
  # utt (e.g. by speed perturbing) they map to the same 'uniq' value.  This is
  # so that we can properly hold out validation data for neural net training and
  # know that we're not training on perturbed verions of that utterance.  We
  # need to obtain the utt2uniq file so that if any 2 'new' utts contain any of
  # the same 'old' utts, their 'uniq' values are the same [but otherwise as far
  # as possible, the 'uniq' values are different.]
  #
  # we'll do this by arranging the old 'uniq' values into groups as necessary to
  # capture this property.

  # The following command creates 'uniq_sets', each line of which contains
  # a set of original 'uniq' values, and effectively we assert that they must
  # be grouped together to the same 'uniq' value.
  # the first awk command prints a group of the original utterance-ids that
  # are combined together into a single new utterance, and the apply_map
  # command converts those into a list of original 'uniq' values.
  awk '{$1 = ""; print;}' < $dir/utt2utts | \
    utils/apply_map.pl $srcdir/utt2uniq > $dir/uniq_sets

  # The next command creates $dir/uniq2merged_uniq, which is a map from the
  # original 'uniq' values to the 'merged' uniq values.
  # for example, if $dir/uniq_sets were to contain
  # a b
  # b c
  # d
  # then we'd obtain a uniq2merged_uniq file that looks like:
  # a a
  # b a
  # c a
  # d d
  # ... because a and b appear together, and b and c appear together,
  # they have to be merged into the same set, and we name that set 'a'
  # (in general, we take the lowest string in lexicographical order).

  cat $dir/uniq_sets | LC_ALL=C python -c '
import sys;
from collections import defaultdict
uniq2orig_uniq = dict()
equal_pairs = set()  # set of 2-tuples (a,b) which should have equal orig_uniq
while True:
    line = sys.stdin.readline()
    if line == "": break
    split_line = line.split() # list of uniq strings that should map in same set
    # initialize uniq2orig_uniq to the identity mapping
    for uniq in split_line: uniq2orig_uniq[uniq] = uniq
    for a in split_line[1:]: equal_pairs.add((split_line[0], a))

changed = True
while changed:
    changed = False
    for a,b in equal_pairs:
         min_orig_uniq = min(uniq2orig_uniq[a], uniq2orig_uniq[b])
         for x in [a,b]:
             if uniq2orig_uniq[x] != min_orig_uniq:
                 uniq2orig_uniq[x] = min_orig_uniq
                 changed = True

for uniq in sorted(uniq2orig_uniq.keys()):
    print uniq, uniq2orig_uniq[uniq]
' > $dir/uniq_to_orig_uniq
  rm $dir/uniq_sets


  # In the following command, suppose we have a line like:
  # utt1-comb2 utt1 utt2
  # .. the first awk command retains only the first original utt, to give
  # utt1-comb2 utt1
  # [we can pick one arbitrarily since we know any of them would map to the same
  # orig_uniq value.]
  # the first apply_map.pl command maps the 'utt1' to the 'uniq' value it mapped to
  # in $srcdir, and the second apply_map.pl command maps it to the grouped 'uniq'
  # value obtained by the inline python script above.
  awk '{print $1, $2}' < $dir/utt2utts | utils/apply_map.pl -f 2 $srcdir/utt2uniq | \
    utils/apply_map.pl -f 2 $dir/uniq_to_orig_uniq > $dir/utt2uniq
  rm $dir/uniq_to_orig_uniq
fi

# note: the user will have to recompute the cmvn, as the speakers may have changed.
rm $dir/cmvn.scp 2>/dev/null || true

utils/validate_data_dir.sh --no-wav $dir

if $cleanup; then
  rm $dir/utt2utts
fi
