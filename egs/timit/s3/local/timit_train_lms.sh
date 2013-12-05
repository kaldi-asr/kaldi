#!/bin/bash
# Copyright 2012  Navdeep Jaitly

# Derived from swbd/s3/local/swbd_p1_train_lms.sh scripts. 

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


# To be run from one directory above this script.
# This script takes no arguments.  It assumes you have already run
# timit_data_prep.sh.  
# It takes as input the file
#     [argument 1]/train_trans.txt
# and uses it to create the lexicon (just the phones) and the biphone language model.
# Creates folder [argument 1]/lm

if [ $# != 1 ]; then
  echo "Usage: ../../local/timit_train_lms.sh [data path]"
  echo "eg: ../../local/timit_train_lms.sh data/local"
  exit 1; 
fi 


dir=$1/lm
trans_file=$1/train_trans.txt
phones_file=$1/phones.txt 
lex_file=$1/lexicon.txt

if [ ! -e $trans_file ]; then 
   echo "Transcript file $trans_file not found. Did you run local/timit_data_prep.sh"
   exit 1;
fi

mkdir -p $dir
export LC_ALL=C # You'll get errors about things being not sorted, if you
# have a different locale.
export PATH=$PATH:`pwd`/../../../tools/kaldi_lm
( # First make sure the kaldi_lm toolkit is installed.
 cd ../../../tools || exit 1;
 if [ -d kaldi_lm ]; then
   echo Not installing the kaldi_lm toolkit since it is already there.
 else
   echo Downloading and installing the kaldi_lm tools
   if [ ! -f kaldi_lm.tar.gz ]; then
     wget http://www.danielpovey.com/files/kaldi/kaldi_lm.tar.gz || exit 1;
   fi
   tar -xvzf kaldi_lm.tar.gz || exit 1;
   cd kaldi_lm
   make || exit 1;
   echo Done making the kaldi_lm tools
 fi
) || exit 1;

mkdir -p $dir

echo "Creating phones file, and monophone lexicon (mapping phones to itself)."
cat $trans_file | awk '{for(n=2;n<=NF;n++) print $n; }' | sort | uniq | awk '{print tolower($1) ; }' > $phones_file
cat $phones_file | awk '{print toupper($1) " " $1 ; }' > $lex_file
cat $trans_file | awk '{for(n=2;n<=NF;n++) print $n; }' | sort | uniq -c | \
   sort -nr > $dir/word.counts


# Get counts from acoustic training transcripts, and add  one-count
# for each word in the lexicon. 
cat $trans_file | awk '{for(n=2;n<=NF;n++) print $n; }' | \
  cat - <(cat $lex_file | awk '{print $1}') | \
   sort | uniq -c | sort -nr > $dir/unigram.counts

# note: we probably won't really make use of <UNK> as there aren't any OOVs
cat $dir/unigram.counts  | awk '{print $2}' | local/get_word_map.pl "<s>" "</s>" > $dir/word_map

# note: ignore 2nd field of train.txt, it's the utterance-id.
cat $trans_file | awk -v wmap=$dir/word_map 'BEGIN{while((getline<wmap)>0)map[$1]=$2;}
  { for(n=2;n<=NF;n++) { printf map[$n]; if(n<NF){ printf " "; } else { print ""; }}}' | gzip -c >$dir/train.gz

! merge_ngrams </dev/null >&/dev/null  && \
     echo merge_ngrams not found in kaldi_lm. You need to have kaldi_lm on your path OR && \
     echo You can do the following:  && \
     echo  1. Install the latest version from http://www.danielpovey.com/files/kaldi/kaldi_lm.tar.gz  && \
     echo  2. you delete kaldi_lm, and kaldi_lm.tar.gz in the tools folder. This script will automatically install it. && \
   exit 1;

echo "Creating biphone model"
local/create_biphone_lm.sh  $dir
