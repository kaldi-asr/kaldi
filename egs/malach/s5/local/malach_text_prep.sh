#!/usr/bin/env bash

# Copyright 2019  IBM Corp. (Author: Michael Picheny) Adapted AMI recipe to MALACH corpus
# Copyright 2015, Brno University of Technology (Author: Karel Vesely)
# Copyright 2014, University of Edinburgh (Author: Pawel Swietojanski), 2014, Apache 2.0

if [ $# -ne 1 ]; then
  echo "Usage: $0 <malach-dir>"
  echo " <malach-dir> is download space."
  exit 1;
fi

set -eux

dir=$1

malach_train_stm=$dir/malach.kaldi_lm.v2.stm
malach_dev_stm=$dir/malach.minitest.try3.v2.stm

echo "Creating annotations directory..."

logdir=data/local/downloads; mkdir -p $logdir/log

if [ ! -d $dir/annotations ]; then
  mkdir -p $dir/annotations
fi

wdir=data/local/annotations

if [ ! -d $wdir ]; then
  mkdir -p $wdir
fi

# make final train/dev splits
for dset in train dev; do
    file_variable=malach_${dset}_stm
    eval file_name=\$$file_variable
    sed "s/<>//" $file_name  | awk '{for (i = 1; i <= 5; ++i) printf "%s ", $i; for (i=6; i<=NF; ++i) printf "%s ", toupper($i); printf "\n"}' > $wdir/$dset.txt
done


