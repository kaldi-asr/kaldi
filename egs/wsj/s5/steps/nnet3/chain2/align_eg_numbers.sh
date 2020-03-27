#!/bin/bash

# Copyright   2020 Xiaomi Corporation, Beijing, China(Author: Haowen Qiu).  
# Apache 2.0.
#
# This script manipulates .scp files in the input folder to make sure 
# they have the same number of lines. It appends lines to those .scp 
# files with less number of lines by copying lines randomly from all 
# of other files.

# Begin configuration section.
cmd=run.pl

srand=0
stage=0

echo "$0 $*"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 2 ]; then
  echo "Usage: $0 [opts] <scp-dir> <output-dir>"
  echo " e.g.: $0 exp/chain/tdnn1a_sp/merged_egs exp/chain/tdnn1a_sp/egs"
  echo ""
  exit 1;
fi

scp_dir=$1
dir=$2

# die on error or undefined variable.
set -e -u

maximum_num_lines=$(wc -l $scp_dir/*.scp | head -n -1 | sort -nr | head -n 1 | awk '{print $1}')
echo "maximum num lines is $maximum_num_lines"

cat $scp_dir/*.scp > $dir/.tmp
for file in $scp_dir/*.scp; do
    dst_file=$dir/$(basename "$file")
    cp $file $dst_file 
    num_lines=$(wc -l $file | awk '{print $1}') 
    for ((i=$num_lines;i<$maximum_num_lines;i++)) do
        shuf -n 1 $dir/.tmp >> $dst_file
    done
done
rm $dir/.tmp

wait;
echo "$0: Finished aligning scp file"

