#!/usr/bin/env bash

src=$1
dst=$2

# Select a very small set for testing
utils/subset_data_dir.sh --shortest $src 10 $dst

# make fake transcripts as negative examples
cp $dst/text $dst/text.ori
sed -i "s/ THERE / THOSE /" $dst/text
sed -i "s/ IN / ON /" $dst/text
