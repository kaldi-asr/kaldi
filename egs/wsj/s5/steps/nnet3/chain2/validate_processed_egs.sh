#!/bin/bash

# Copyright   2019  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
# Copyright   2019  Idiap Research Institute (Author: Srikanth Madikeri).  Apache 2.0.
#
# This script validates a directory containing 'processed' egs for 'chain'
# training, i.e. the output of process_egs.sh.  It also helps to document the
# expectations on such a directory.


if [ -f path.sh ]; then . ./path.sh; fi


if [ $# != 1 ]; then
  echo "Usage: $0  <processed-egs-dir>"
  echo " e.g.: $0 exp/chain/tdnn1a_sp/processed_egs"
  echo ""
  echo "Validates that the processed-egs dir has the expected format"
fi

dir=$1

# Note: the .ark files are not actually consumed directly downstream (only via
# the top-level .scp files), but we check them anyway for now.
for f in $dir/train.scp $dir/info.txt \
         $dir/heldout_subset.{ark,scp} $dir/train_subset.{ark,scp} \
         $dir/train.1.scp $dir/train.1.ark; do
  if ! [ -f $f -a -s $f ]; then
    echo "$0: expected file $f to exist and be nonempty."
    exit 1
  fi
done


if [ $(awk '/^dir_type/ { print $2; }' <$dir/info.txt) != "processed_chain_egs" ]; then
  grep dir_type $dir/info.txt
  echo "$0: dir_type should be processed_chain_egs in $dir/info.txt"
  exit 1
fi

lang=$(awk '/^lang / {print $2; }' <$dir/info.txt)

for f in $dir/misc/$lang.{trans_mdl,normalization.fst,den.fst}; do
  if ! [ -f $f -a -s $f ]; then
    echo "$0: expected file $f to exist and be nonempty."
    exit 1
  fi
done

echo "$0: sucessfully validated processed egs in $dir"
