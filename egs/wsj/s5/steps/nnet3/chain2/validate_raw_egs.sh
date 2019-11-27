#!/bin/bash

# Copyright   2019  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
# Copyright   2019  Idiap Research Institute (Author: Srikanth Madikeri).  Apache 2.0.
#
# This script validates a directory containing 'raw' egs for 'chain' training.
# It also helps to document the expectations on such a directory.



if [ -f path.sh ]; then . ./path.sh; fi


if [ $# != 1 ]; then
  echo "Usage: $0  <raw-egs-dir>"
  echo " e.g.: $0 exp/chaina/tdnn1a_sp/raw_egs"
  echo ""
  echo "Validates that the raw-egs dir has the expected format"
fi

dir=$1

for f in $dir/all.scp $dir/cegs.1.ark $dir/info.txt \
         $dir/misc/utt2spk; do
  if ! [ -s $f ]; then
    echo "$0: expected file $f to exist and be nonempty."
    exit 1
  fi
done


if [ $(awk '/^dir_type/ { print $2; }' <$dir/info.txt) != "raw_chain_egs" ]; then
  grep dir_type $dir/info.txt
  echo "$0: dir_type should be raw_chain_egs in $dir/info.txt"
  exit 1
fi

lang=$(awk '/^lang / {print $2; }' <$dir/info.txt)

for f in $dir/misc/$lang.{trans_mdl,normalization.fst,den.fst}; do
  if ! [ -s $f ]; then
    echo "$0: expected file $f to exist and be nonempty."
    exit 1
  fi
done

echo "$0: sucessfully validated raw egs in $dir"
