#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)
# Apache 2.0.


if [ $# -ne 3 ]; then
   echo "Usage: local/kws_data_prep.sh <lang-dir> <data-dir> <kws-data-dir>"
   echo " e.g.: local/kws_data_prep.sh data/lang_test_bd_tgpr/ data/test_eval92/ data/kws/"
   exit 1;
fi

langdir=$1;
datadir=$2;
kwsdatadir=$3;

mkdir -p $kwsdatadir;

# Create keyword id for each keyword
cat $kwsdatadir/raw_keywords.txt | perl -e '
  $idx=1;
  while(<>) {
    chomp;
    printf "WSJ-%04d $_\n", $idx;
    $idx++;
  }' > $kwsdatadir/keywords.txt

# Map the keywords to integers; note that we remove the keywords that
# are not in our $langdir/words.txt, as we won't find them anyway...
cat $kwsdatadir/keywords.txt | \
  sym2int.pl --map-oov 0 -f 2- $langdir/words.txt | \
  grep -v " 0 " | grep -v " 0$" > $kwsdatadir/keywords.int

# Compile keywords into FSTs
transcripts-to-fsts ark:$kwsdatadir/keywords.int ark:$kwsdatadir/keywords.fsts

# Create utterance id for each utterance; Note that by "utterance" here I mean
# the keys that will appear in the lattice archive. You may have to modify here
cat $datadir/wav.scp | \
  awk '{print $1}' | \
  sort | uniq | perl -e '
  $idx=1;
  while(<>) {
    chomp;
    print "$_ $idx\n";
    $idx++;
  }' > $kwsdatadir/utter_id

# Map utterance to the names that will appear in the rttm file. You have 
# to modify the commands below accoring to your rttm file. In the WSJ case
# since each file is an utterance, we assume that the actual file names will 
# be the "names" in the rttm, so the utterance names map to themselves.
cat $datadir/wav.scp | \
  awk '{print $1}' | \
  sort | uniq | perl -e '
  while(<>) {
    chomp;
    print "$_ $_\n";
  }' > $kwsdatadir/utter_map;
echo "Kws data preparation succeeded"
