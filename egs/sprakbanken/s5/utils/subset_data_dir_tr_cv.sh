#!/bin/bash
# Copyright 2013  Hong Kong University of Science and Technology (Author: Ricky Chan Ho Yin);
#                 Brno University of Technology (Author: Karel Vesely);
#                 Johns Hopkins University (Author: Daniel Povey);
# Apache 2.0

# This script splits dataset to two parts : 
# training set from (100-P)% of speakers/utterances and 
# held-out set (or cross-validation) from P% of remaining speakers/remaining utterances,
# which will be later on used for neural network training
#
# There are two options for choosing held-out (or cross-validation) set, either by
# --cv-spk-percent P , which will give you CV set based on random chosen P% of speakers, or
# --cv-utt-percent P , which will give you CV set based on last P% utterances in the dataset
# 
# If you don't apply the above two options, by default the script will use --cv-utt-percent option,
# and the default cross validation percentage portion is equal to 10% (i.e. P=10)
#
# The --cv-spk-percent option is useful if you would like to have subset chosen from random speakers order, 
# especially for the cases where dataset contains multiple different corpora,
# where type of speakers or recording channels may be quite different 

# Begin configuration.
cv_spk_percent= # % of speakers is parsed by option
cv_utt_percent=10 # default 10% of total utterances 
seed=777 # use seed for speaker shuffling
# End configuration.

echo "$0 $@"  # Print the command line for logging

uttbase=true; # by default, we choose last 10% utterances for CV

if [ "$1" == "--cv-spk-percent" ]; then
  uttbase=false;
  spkbase=true;
fi

[ -f path.sh ] && . ./path.sh; 

. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 [--cv-spk-percent P|--cv-utt-percent P] <srcdir> <traindir> <crossvaldir>"
  echo "  --cv-spk-percent P  Cross Validation portion of the total speakers, recommend value is 10% (i.e. P=10)"
  echo "  --cv-utt-percent P  Cross Validation portion of the total utterances, default is 10% (i.e. P=10)"
  echo "  "
  exit 1;
fi

srcdir=$1
trndir=$2
cvdir=$3

## use simple last P% utterance for CV
if $uttbase; then
  if [ ! -f $srcdir/utt2spk ]; then
    echo "$0: no such file $srcdir/utt2spk"
    exit 1;
  fi

  #total number of lines
  N=$(cat $srcdir/utt2spk | wc -l)
  #get line number where (100-P)% of the data lies
  P_utt=$((N * cv_utt_percent / 100))
  N_head=$((N -P_utt))
  #move the boundary so it is located on speaker change
  N_head=$(cat $srcdir/utt2spk | uniq -f1 -c | awk '{ if(n+$1<='$N_head') { n += $1 } else { nextfile } } END{ print n }')
  #the rest of the data will be that big
  N_tail=$((N-N_head))

  #now call the subset_data_dir.sh and fix the directories
  subset_data_dir.sh --first $srcdir $N_head $trndir
  subset_data_dir.sh --last $srcdir $N_tail $cvdir

  exit 0;
fi

## use random chosen P% speakers for CV
if [ ! -f $srcdir/spk2utt ]; then
  echo "$0: no such file $srcdir/spk2utt" 
  exit 1;
fi

#total, cv, train number of speakers
N=$(cat $srcdir/spk2utt | wc -l)
N_spk_cv=$((N * cv_spk_percent / 100))
N_spk_trn=$((N - N_spk_cv))

mkdir -p $cvdir $trndir

#shuffle the speaker list
awk '{print $1}' $srcdir/spk2utt | shuffle_list.pl --srand $seed > $trndir/_tmpf_randspk

#split the train/cv
head -n $N_spk_cv $trndir/_tmpf_randspk > $cvdir/_tmpf_cvspk
tail -n $N_spk_trn $trndir/_tmpf_randspk > $trndir/_tmpf_trainspk

#now call the subset_data_dir.sh 
subset_data_dir.sh --spk-list $trndir/_tmpf_trainspk $srcdir $trndir
subset_data_dir.sh --spk-list $cvdir/_tmpf_cvspk $srcdir $cvdir

#clean-up
rm -f $trndir/_tmpf_randspk $trndir/_tmpf_trainspk $cvdir/_tmpf_cvspk

