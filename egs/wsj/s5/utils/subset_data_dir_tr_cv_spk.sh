#!/bin/bash
# Copyright 2013  Hong Kong University of Science and Technology (Author: Ricky Chan Ho Yin);
#                 Karel Vesely, Daniel Povey;
# Apache 2.0

# This script splits dataset to two parts : 
# training set from (100-P)% of speakers and 
# held-out set (or cross-validation) from P% of remaining speakers,
# which will be later on used for neural network training
# The default cross validation percentage portion is 10% (i.e. P=10)
#
# It is useful if you would like to have subset chosen from random speakers order, 
# especially for the cases where a dataset contains multiple different corpora, where  
# type of speakers or recording channels may be quite different 

if [ $# != 3 ] && [ $# != 5 ]; then
  echo "Usage: $0 <srcdir> <traindir> <crossvaldir> [--cvportion P]"
  echo "--cvportion P        Cross Validation portion of the total speakers, default is 10% (i.e. P=10)"
  exit 1;
fi

srcdir=$1
trndir=$2
cvdir=$3

if [ $# == 5 ]; then
  if [ $4 = "--cvportion" ]; then
    cvportion=$5
  else
    echo "Usage: $0 <srcdir> <traindir> <crossvaldir> [--cvportion P]"
    echo "--cvportion P        Cross Validation portion of the total speakers, default is 10% (i.e. P=10)"
    exit 1;
  fi
else 
  cvportion=10
fi

if [ ! -f $srcdir/spk2utt ]; then
  echo "$0: no such file $srcdir/spk2utt" 
  exit 1;
fi

#total number of lines
N=$(cat $srcdir/spk2utt | wc -l)
awk '{print $1}' $srcdir/spk2utt |  awk 'BEGIN{srand();}{print rand()"\t"$0}' | sort -k1 -n | cut -f2- > $srcdir/_tmpf_randspk
boundary=$((N*cvportion/100)) 
tailboundary=$((N-$boundary))

mkdir -p $cvdir $trndir
head -$boundary $srcdir/_tmpf_randspk > $cvdir/_tmpf_cvspk
tail -$tailboundary $srcdir/_tmpf_randspk > $trndir/_tmpf_trainspk

#now call the subset_data_dir.sh 
subset_data_dir.sh --spk-list $trndir/_tmpf_trainspk $srcdir $trndir
subset_data_dir.sh --spk-list $cvdir/_tmpf_cvspk $srcdir $cvdir

rm -f $srcdir/_tmpf_randspk $trndir/_tmpf_trainspk $cvdir/_tmpf_cvspk

