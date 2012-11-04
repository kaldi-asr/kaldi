#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)
# Apache 2.0.


if [ $# -ne 4 ]; then
   echo "Usage: local/kws_data_prep.sh keyword_list <lang-dir> <data-dir> <kws-data-dir>"
   echo " e.g.: local/kws_data_prep.sh keywords data/lang/ data/eval/ data/kws/"
   exit 1;
fi

keywords=$1;
langdir=$2;
datadir=$3;
kwsdatadir=$4;

mkdir -p $kwsdatadir;

# This script is an example for the Babel Cantonese STD task

# Create keyword id for each keyword
cat $keywords | perl -e '
  $idx=1;
  while(<>) {
    chomp;
    printf "KW101-%04d $_\n", $idx;
    $idx++;
  }' > $kwsdatadir/keywords.txt

# Map the keywords to integers; note that we remove the keywords that
# are not in our $langdir/words.txt, as we won't find them anyway...
cat $kwsdatadir/keywords.txt | \
  sym2int.pl --map-oov 0 -f 2- $langdir/words.txt | \
  grep -v " 0 " | grep -v " 0$" > $kwsdatadir/keywords.int

# Compile keywords into FSTs
transcripts-to-fsts ark:$kwsdatadir/keywords.int ark:$kwsdatadir/keywords.fsts

# Create utterance id for each utterance
cat $datadir/segments | \
  awk '{print $1}' | \
  sort | uniq | perl -e '
  $idx=1;
  while(<>) {
    chomp;
    print "$_ $idx\n";
    $idx++;
  }' > $kwsdatadir/utter_id

# Map utterance to the names that will appear in the rttm file. You have 
# to modify the commands below accoring to your rttm file
cat $datadir/segments | \
  awk '{print $1}' | \
  sort | uniq | perl -e '
  while(<>) {
    chomp;
    print "$_ ";
    s/_[0-9]{6}$//g;
    if (m/_inLine/) {
      s/_inLine//g;
      $_.="_inLine";
    } else {
      s/_outLine//g;
      $_.="_outLine";
    }
    print "$_\n";
  }' > $kwsdatadir/utter_map;
echo "Kws data preparation succeeded"
