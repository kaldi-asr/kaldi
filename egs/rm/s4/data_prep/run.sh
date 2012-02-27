#!/bin/bash
# This script should be run from the directory where it is located (i.e. data_prep)
# Copyright 2010-2011 Microsoft Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

echo "--- Starting data preparation ..."

if [ $# != 1 ]; then
   echo "Usage: ./run.sh /path/to/RM"
   exit 1; 
fi 

RMROOT=$1
if [ ! -d $RMROOT/LDC93S3B -o ! -d $RMROOT/rm1 ]; then
  echo "Speech data is missing. Please run getdata.sh"
  exit 1; 
fi  

# Make a list of files
cat $RMROOT/rm1/etc/rm1_train.fileids | \
    xargs -I_x_ echo $RMROOT/rm1/feat/_x_.mfc > train.flist
cat $RMROOT/rm1/etc/rm1_test.fileids | \
    xargs -I_x_ echo $RMROOT/rm1/feat/_x_.mfc > test.flist

# make_trans.pl also creates the utterance id's and the kaldi-format scp file.

# train
./make_trans.pl trn train.flist $RMROOT/LDC93S3B/disc_1/doc/al_sents.snr train_trans.txt train.scp
mv train_trans.txt tmp; sort -k 1 tmp > train_trans.txt
mv train.scp tmp; sort -k 1 tmp > train.scp

# test
./make_trans.pl test test.flist $RMROOT/LDC93S3B/disc_1/doc/al_sents.snr test_trans.txt test.scp
mv test_trans.txt tmp; sort -k 1 tmp > test_trans.txt
mv test.scp tmp; sort -k 1 tmp > test.scp

# We already have the features, so sph2pipe step isn't needed

../scripts/make_rm_lm.pl $RMROOT/LDC93S3B/disc_1/doc/wp_gram.txt  > G.txt 

# Convert the CMU's lexicon to a form which the other scripts expect
# (leave only the first pronunciation variant, convert "'" to "+", 
# and convert the phones to lower case)
cat $RMROOT/rm1/etc/rm1.dic | \
  egrep -v '\(' | \
  sed -e "s/'/\+/g" | \
  sed -e "s/^\([[:alnum:]-]\+\(+[[:alpha:]]\+\)\?\)\(.*\)/\1\L\3/g" > lexicon.txt 

echo "--- Done data preparation!"
