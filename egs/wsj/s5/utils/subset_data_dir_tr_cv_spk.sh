#!/bin/bash
# Copyright 2013  Hong Kong University of Science and Technology (Author: Ricky Chan Ho Yin);
#                 Brno University of Technology (Author: Karel Vesely);
#                 Johns Hopkins University (Author: Daniel Povey);
# Apache 2.0

# This script splits dataset to two parts : 
# training set from (100-P)% of speakers and 
# held-out set (or cross-validation) from P% of remaining speakers,
# which will be later on used for neural network training
# The default cross validation percentage portion is 10% (i.e. P=10)
#
# It is useful if you would like to have subset chosen from random speakers order, 
# especially for the cases where a dataset contains multiple different corpora,
# where type of speakers or recording channels may be quite different 

# Begin configuration.
cv_spk_percent=10 # default 10% of speakers
seed=777 # use seed for speaker shuffling
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 

. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 [--cv-spk-percent P] <srcdir> <traindir> <crossvaldir>"
  echo "  --cv-spk-percent P  Cross Validation portion of the total speakers, default is 10% (i.e. P=10)"
  exit 1;
fi

srcdir=$1
trndir=$2
cvdir=$3

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

