#!/bin/bash

# Copyright 2017  Hossein Hadian

# Apache 2.0

write_utt2orig=              # if provided, this script will write
                             # a mapping of shifted utterance ids
                             # to the original ones into the file
                             # specified by this option

echo "$0 $@"  # Print the command line for logging
if [ -f path.sh ]; then . ./path.sh; fi
. utils/parse_options.sh

if [ $# != 3 ]; then
  echo "Usage: $0 <frame-subsampling-factor> <srcdir> <destdir>"
  echo "e.g.: $0 3 data/train data/train_fs3"
  echo "For use in perturbing data for discriminative training and alignment of"
  echo "frame-subsampled systems, this script uses utils/data/shift_feats.sh"
  echo "and utils/data/combine_data.sh to shift the features"
  echo "<frame-subsampling-factor> different ways and combine them."
  echo "E.g. if <frame-subsampling-factor> is 3, this script will combine"
  echo "the data frame-shifted by -1, 0 and 1 (c.f. shift-feats)."
  exit 1
fi

frame_subsampling_factor=$1
srcdir=$2
destdir=$3

if [ ! -f $srcdir/feats.scp ]; then
  echo "$0: expected $srcdir/feats.scp to exist"
  exit 1
fi

if [ -f $destdir/feats.scp ]; then
  echo "$0: $destdir/feats.scp already exists: refusing to run this (please delete $destdir/feats.scp if you want this to run)"
  exit 1
fi

if [ ! -z $write_utt2orig ]; then
  awk '{print $1 " " $1}' $srcdir/feats.scp >$write_utt2orig
fi

tmp_shift_destdirs=()
for frame_shift in `seq $[-(frame_subsampling_factor/2)] $[-(frame_subsampling_factor/2) + frame_subsampling_factor - 1]`; do
  if [ "$frame_shift" == 0 ]; then continue; fi
  utils/data/shift_feats.sh $frame_shift $srcdir ${destdir}_fs$frame_shift || exit 1
  tmp_shift_destdirs+=("${destdir}_fs$frame_shift")
  if [ ! -z $write_utt2orig ]; then
    awk -v prefix="fs$frame_shift-" '{printf("%s%s %s\n", prefix, $1, $1);}' $srcdir/feats.scp >>$write_utt2orig
  fi  
done
utils/data/combine_data.sh $destdir $srcdir ${tmp_shift_destdirs[@]} || exit 1
rm -r ${tmp_shift_destdirs[@]}

utils/validate_data_dir.sh $destdir

src_nf=`cat $srcdir/feats.scp | wc -l`
dest_nf=`cat $destdir/feats.scp | wc -l`
if [ $[src_nf*frame_subsampling_factor] -ne $dest_nf ]; then
  echo "There was a problem. Expected number of feature lines in destination dir to be $[src_nf*frame_subsampling_factor];"
  exit 1;
fi

echo "$0: Successfully generated $frame_subsampling_factor-way shifted version of data in $srcdir, in $destdir"
