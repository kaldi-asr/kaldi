#!/usr/bin/env bash

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

# Generate keywords; we generate 20 unigram keywords with at least 20 counts,
# 20 bigram keywords with at least 10 counts and 10 trigram keywords with at
# least 5 counts.
cat $text | perl -e '
  %unigram = ();
  %bigram = ();
  %trigram = ();
  while(<>) {
    chomp;
    @col=split(" ", $_);
    shift @col;
    for($i = 0; $i < @col; $i++) {
      # unigram case
      if (!defined($unigram{$col[$i]})) {
        $unigram{$col[$i]} = 0;
      }
      $unigram{$col[$i]}++;

      # bigram case
      if ($i < @col-1) {
        $word = $col[$i] . " " . $col[$i+1];
        if (!defined($bigram{$word})) {
          $bigram{$word} = 0;
        }
        $bigram{$word}++;
      }

      # trigram case
      if ($i < @col-2) {
        $word = $col[$i] . " " . $col[$i+1] . " " . $col[$i+2];
        if (!defined($trigram{$word})) {
          $trigram{$word} = 0;
        }
        $trigram{$word}++;
      }
    }
  }

  $max_count = 100;
  $total = 20;
  $current = 0;
  $min_count = 20;
  while ($current < $total && $min_count <= $max_count) {
    foreach $x (keys %unigram) {
      if ($unigram{$x} == $min_count) {
        print "$x\n";
        $unigram{$x} = 0;
        $current++;
      }
      if ($current == $total) {
        last;
      }
    }
    $min_count++;
  }
  
  $total = 20;
  $current = 0;
  $min_count = 4;
  while ($current < $total && $min_count <= $max_count) {
    foreach $x (keys %bigram) {
      if ($bigram{$x} == $min_count) {
        print "$x\n";
        $bigram{$x} = 0;
        $current++;
      }
      if ($current == $total) {
        last;
      }
    }
    $min_count++;
  }
  
  $total = 10;
  $current = 0;
  $min_count = 3;
  while ($current < $total && $min_count <= $max_count) {
    foreach $x (keys %trigram) {
      if ($trigram{$x} == $min_count) {
        print "$x\n";
        $trigram{$x} = 0;
        $current++;
      }
      if ($current == $total) {
        last;
      }
    }
    $min_count++;
  }
  ' > $kwsdatadir/raw_keywords.txt

echo "Keywords generation succeeded"
