#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)
# Apache 2.0.


if [ $# -ne 2 ]; then
   echo "Usage: local/generate_example_kws.sh <data-dir> <kws-data-dir>"
   echo " e.g.: local/generate_example_kws.sh data/test_eval92/ <data/kws>"
   exit 1;
fi

datadir=$1;
kwsdatadir=$2;
text=$datadir/text;

mkdir -p $kwsdatadir;

# Generate keywords; we randomly generate 100 keywords with at least 5 counts
# in text 
cat $text | perl -e '
  %counts = ();
  while(<>) {
    chomp;
    @col=split(" ", $_);
    shift @col;
    foreach $x (@col) {
      if (!defined($counts{$x})) {
        $counts{$x} = 1;
      } else {
        $counts{$x} = $counts{$x}+1;
      }
    }
  }

  $total = 50;
  $current = 0;
  $print_count = 15;
  while ($current < $total) {
    foreach $x (keys %counts) {
      if ($counts{$x} == $print_count) {
        print "$x\n";
        $counts{$x} = 0;
        $current++;
      }
      if ($current == $total) {
        last;
      }
    }
    $print_count++;
  }' > $kwsdatadir/keywords.txt

echo "Keywords generation succeeded"
