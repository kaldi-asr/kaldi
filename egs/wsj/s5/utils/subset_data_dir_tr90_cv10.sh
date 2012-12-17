#!/bin/bash
# Copyright 2010-2012  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This script splits dataset to two parts : 
# 90% training set and 10% held-out set (or cross-validation),
# which will be later on used for neural network training
#
# It is useful if the database is not presplit or where
# we cannot get alignment on dev set

if [ $# != 3 ]; then
  echo "Usage: $0 <srcdir> <traindir> <crossvaldir>"
  exit 1;
fi

srcdir=$1
trndir=$2
cvdir=$3

if [ ! -f $srcdir/utt2spk ]; then
  echo "$0: no such file $srcdir/utt2spk" 
  exit 1;
fi

#total number of lines
N=$(cat $srcdir/utt2spk | wc -l)
#get line number where 90% of the data lies
N_head=$((N*9/10)) 
#move the boundary so it is located on speaker change
N_head=$(cat $srcdir/utt2spk | uniq -f1 -c | awk '{ if(n+$1<='$N_head') { n += $1 } else { nextfile } } END{ print n }')
#the rest of the data will be that big
N_tail=$((N-N_head))

#now call the subset_data_dir.sh and fix the directories
subset_data_dir.sh --first $srcdir $N_head $trndir
subset_data_dir.sh --last $srcdir $N_tail $cvdir
