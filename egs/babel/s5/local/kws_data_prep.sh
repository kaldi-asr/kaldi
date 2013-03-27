#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)
# Apache 2.0.

# Begin configuration section.  
case_insensitive=true
# End configuration section.

echo $0 "$@"

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;


if [ $# -ne 3 ]; then
   echo "Usage: local/kws_data_prep.sh <lang-dir> <data-dir> <kws-data-dir>"
   echo " e.g.: local/kws_data_prep.sh data/lang/ data/eval/ data/kws/"
   exit 1;
fi

langdir=$1;
datadir=$2;
kwsdatadir=$3;
keywords=$kwsdatadir/kwlist.xml


mkdir -p $kwsdatadir;

# This script is an example for the Babel Cantonese STD task

cat $keywords | perl -e '
  #binmode STDIN, ":utf8"; 
  binmode STDOUT, ":utf8"; 

  use XML::Simple;
  use Data::Dumper;

  my $data = XMLin(\*STDIN);

  #print Dumper($data->{kw});
  foreach $kwentry (@{$data->{kw}}) {
    #print Dumper($kwentry);
    print "$kwentry->{kwid}\t$kwentry->{kwtext}\n";
  }
' > $kwsdatadir/keywords.txt


# Map the keywords to integers; note that we remove the keywords that
# are not in our $langdir/words.txt, as we won't find them anyway...
#cat $kwsdatadir/keywords.txt | babel/filter_keywords.pl $langdir/words.txt - - | \
#  sym2int.pl --map-oov 0 -f 2- $langdir/words.txt | \
if $case_insensitive  ; then
  echo "Running case insensitive processing"
  cat $langdir/words.txt | tr '[:lower:]' '[:upper:]'  > $kwsdatadir/words.txt
  [ `cut -f 1 -d ' ' $kwsdatadir/words.txt | sort -u | wc -l` -ne `cat $kwsdatadir/words.txt | wc -l` ] && echo "Warning, multiple words in dictionary differ only in case..."

  cat $kwsdatadir/keywords.txt | tr '[:lower:]' '[:upper:]'  | \
    sym2int.pl --map-oov 0 -f 2- $kwsdatadir/words.txt > $kwsdatadir/keywords_all.int
else
  cp $langdir/words.txt  $kwsdatadir/words.txt
  cat $kwsdatadir/keywords.txt | \
    sym2int.pl --map-oov 0 -f 2- $kwsdatadir/words.txt > $kwsdatadir/keywords_all.int
fi

cat $kwsdatadir/keywords_all.int | \
  grep -v " 0 " | grep -v " 0$" > $kwsdatadir/keywords.int

cut -f 1 -d ' ' $kwsdatadir/keywords.int | \
  local/subset_kwslist.pl $keywords > $kwsdatadir/kwlist_invocab.xml

cat $kwsdatadir/keywords_all.int | \
  egrep " 0 | 0$" | cut -f 1 -d ' ' | \
  local/subset_kwslist.pl $keywords > $kwsdatadir/kwlist_outvocab.xml



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
cat $datadir/segments | awk '{print $1" "$2}' | sort | uniq > $kwsdatadir/utter_map;

echo "Kws data preparation succeeded"
