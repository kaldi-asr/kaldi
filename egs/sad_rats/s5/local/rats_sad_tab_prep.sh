#!/usr/bin/env bash

# Copyright 2021 ARL (Author: John Morgan)
# Apache 2.0.

# This script consolidates the information about each recording.
# The input is the path to the directory where the tabular files associated with
# each recording are stored.
# There are 3 output files, one for each fold.
# The output files are stored under data/local/annotation.

if [ $# -ne 1 ]; then
  echo "Usage: $0 <rats_sad_dir>"
  echo "<rats_sad_dir>: Source data location"
  echo "For example:"
  echo "$0 /mnt/corpora/LDC2015S02/RATS_SAD/data"
echo "Output is written to data/local/annotations/{train,dev-1,dev-2}."
  exit 1;
fi

dir=$1

mkdir -p data/local/annotations

for fld in train dev-1 dev-2; do
  find $dir/$fld/sad -type f -name "*.tab" | xargs cat > \
    data/local/annotations/$fld.txt
done
