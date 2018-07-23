#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)
# Apache 2.0.

# Begin configuration section.
case_insensitive=true
use_icu=true
icu_transform="Any-Lower"
silence_word=  # Optional silence word to insert (once) between words of the transcript.
# End configuration section.

echo $0 "$@"

help_message="
   Usage: local/kws_data_prep.sh <lang-dir> <data-dir> <kws-data-dir>
    e.g.: local/kws_data_prep.sh data/lang/ data/eval/ data/kws/
   Input is in <kws-data-dir>: kwlist.xml, ecf.xml (rttm file not needed).
   Output is in <kws-data/dir>: keywords.txt, keywords_all.int, kwlist_invocab.xml,
       kwlist_outvocab.xml, keywords.fsts
   Note: most important output is keywords.fsts
   allowed switches:
      --case-sensitive <true|false>      # Shall we be case-sensitive or not?
                                         # Please not the case-sensitivness depends
                                         # on the shell locale!
      --use-uconv <true|false>           # Use the ICU uconv binary to normalize casing
      --icu-transform <string>           # When using ICU, use this transliteration

"

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;


if [ $# -ne 3 ]; then
  printf "FATAL: invalid number of arguments.\n\n"
  printf "$help_message\n"
  exit 1;
fi

set -u
set -e
set -o pipefail

langdir=$1;
datadir=$2;
kwsdatadir=$3;
keywords=$kwsdatadir/kwlist.xml


mkdir -p $kwsdatadir;

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
if  $case_insensitive && ! $use_icu  ; then
  echo "$0: Running case insensitive processing"
  cat $langdir/words.txt | tr '[:lower:]' '[:upper:]'  > $kwsdatadir/words.txt
  [ `cut -f 1 -d ' ' $kwsdatadir/words.txt | sort -u | wc -l` -ne `cat $kwsdatadir/words.txt | wc -l` ] && \
    echo "$0: Warning, multiple words in dictionary differ only in case: "


  cat $kwsdatadir/keywords.txt | tr '[:lower:]' '[:upper:]'  | \
    sym2int.pl --map-oov 0 -f 2- $kwsdatadir/words.txt > $kwsdatadir/keywords_all.int
elif  $case_insensitive && $use_icu ; then
  echo "$0: Running case insensitive processing (using ICU with transform \"$icu_transform\")"
  cat $langdir/words.txt | uconv -f utf8 -t utf8 -x "${icu_transform}"  > $kwsdatadir/words.txt
  [ `cut -f 1 -d ' ' $kwsdatadir/words.txt | sort -u | wc -l` -ne `cat $kwsdatadir/words.txt | wc -l` ] && \
    echo "$0: Warning, multiple words in dictionary differ only in case: "

  paste <(cut -f 1  $kwsdatadir/keywords.txt  ) \
        <(cut -f 2  $kwsdatadir/keywords.txt | uconv -f utf8 -t utf8 -x "${icu_transform}" ) |\
    local/kwords2indices.pl --map-oov 0  $kwsdatadir/words.txt > $kwsdatadir/keywords_all.int
else
  cp $langdir/words.txt  $kwsdatadir/words.txt
  cat $kwsdatadir/keywords.txt | \
    sym2int.pl --map-oov 0 -f 2- $kwsdatadir/words.txt > $kwsdatadir/keywords_all.int
fi

(cat $kwsdatadir/keywords_all.int | \
  grep -v " 0 " | grep -v " 0$" > $kwsdatadir/keywords.int ) || true

(cut -f 1 -d ' ' $kwsdatadir/keywords.int | \
  local/subset_kwslist.pl $keywords > $kwsdatadir/kwlist_invocab.xml) || true

(cat $kwsdatadir/keywords_all.int | \
  egrep " 0 | 0$" | cut -f 1 -d ' ' | \
  local/subset_kwslist.pl $keywords > $kwsdatadir/kwlist_outvocab.xml) || true


# Compile keywords into FSTs
if [ -s $kwsdatadir/keywords.int ]; then
  if [ -z $silence_word ]; then
    transcripts-to-fsts ark:$kwsdatadir/keywords.int ark,t:$kwsdatadir/keywords.fsts
  else
    silence_int=`grep -w $silence_word $langdir/words.txt | awk '{print $2}'`
    [ -z $silence_int ] && \
       echo "$0: Error: could not find integer representation of silence word $silence_word" && exit 1;
    transcripts-to-fsts ark:$kwsdatadir/keywords.int ark,t:- | \
      awk -v 'OFS=\t' -v silint=$silence_int '{if (NF == 4 && $1 != 0) { print $1, $1, silint, silint; } print; }' \
       > $kwsdatadir/keywords.fsts
  fi
else
  echo "WARNING: $kwsdatadir/keywords.int is zero-size. That means no keyword"
  echo "WARNING: was found in the dictionary. That might be OK -- or not."
  touch $kwsdatadir/keywords.fsts
fi

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

echo "$0: Kws data preparation succeeded"
