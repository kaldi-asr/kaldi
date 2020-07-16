#!/usr/bin/env bash

# Copyright 2019 Johns Hopkins University (author: Jinyi Yang)
# Apache 2.0

# This script checks the Out Of Vocabulary words rate of given data set.

if [ $# -ne 2 ]; then
  echo "Usage: $0 <lexicon> <text-file>"
  exit 1
fi
lex=$1
fname=$2

cat $fname | awk '{for(n=2;n<=NF;n++) { print $n; }}' | perl -e '
  $lex = shift @ARGV; open(L, "<$lex")||die;
  while(<L>){ @A=split; $seen{$A[0]}=1;}
  while(<STDIN>) {
    @A=split;
    $word=$A[0];
    $tot++;
    if(defined $seen{$word}) {
      $invoc++;
    } else {print "OOV word $word\n";}
  }
  $oov_rate = 100.0 * (1.0 - ($invoc / $tot));
  printf("Seen $invoc out of $tot tokens; OOV rate is %.2f\n", $oov_rate);
  ' $lex
