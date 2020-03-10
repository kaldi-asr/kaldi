#!/bin/bash

# Copyright   2019  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
# Copyright   2019  Idiap Research Institute (Author: Srikanth Madikeri).  Apache 2.0.
#
# This script validates a directory containing 'randomized' egs for 'chain'
# training, i.e. the output of randomize_egs.sh (this is the final form of the
# egs which is consumed by the training script).  It also helps to document the
# expectations on such a directory.


if [ -f path.sh ]; then . ./path.sh; fi


if [ $# != 1 ]; then
  echo "Usage: $0  <randomized-egs-dir>"
  echo " e.g.: $0 exp/chain/tdnn1a_sp/egs"
  echo ""
  echo "Validates that the final (randomized) egs dir has the expected format"
fi

dir=$1

# Note: the .ark files are not actually consumed directly downstream (only via
# the top-level .scp files), but we check them anyway for now.
for f in $dir/train.1.scp $dir/info.txt \
         $dir/heldout_subset.scp $dir/train_subset.scp; do
  if ! [ -f $f -a -s $f ]; then
    echo "$0: expected file $f to exist and be nonempty."
    exit 1
  fi
done


if [ $(awk '/^dir_type/ { print $2; }' <$dir/info.txt) != "randomized_chain_egs" ]; then
  grep dir_type $dir/info.txt
  echo "$0: dir_type should be randomized_chain_egs in $dir/info.txt"
  exit 1
fi

langs=$(awk '/^langs / {$1 = ""; print; }' <$dir/info.txt)
num_scp_files=$(awk '/^num_scp_files / { print $2; }' <$dir/info.txt)

if [ -z "$langs" ]; then
  echo "$0: expecting the list of languages to be nonempty in $dir/info.txt"
  exit 1
fi

for lang in $langs; do
  for f in $dir/misc/$lang.{trans_mdl,normalization.fst,den.fst} $dir/info_${lang}.txt; do
    if ! [ -f $f -a -s $f ]; then
      echo "$0: expected file $f to exist and be nonempty."
      exit 1
    fi
  done
done

for i in $(seq $num_scp_files); do
  if ! [ -s $dir/train.$i.scp ]; then
    echo "$0: expected file $dir/train.$i.scp to exist and be nonempty."
    exit 1
  fi
done


echo "$0: sucessfully validated randomized egs in $dir"
