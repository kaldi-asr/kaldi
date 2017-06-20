#!/usr/bin/env perl
# Copyright 2016 Alibaba Robotics Corp. (Author: Xingyu Na)
#
# A script for char-based Chinese OOV lexicon generation.
#
# Input 1: char-based dictionary, example
# CHAR1 ph1 ph2
# CHAR2 ph3
# CHAR3 ph2 ph4
#
# Input 2: OOV word list, example
# WORD1
# WORD2
# WORD3
#
# where WORD1 is in the format of "CHAR1CHAR2".
#
# Output: OOV lexicon, in the format of normal lexicon

if($#ARGV != 1) {
  print STDERR "usage: perl create_oov_char_lexicon.pl chardict oovwordlist > oovlex\n\n";
  print STDERR "### chardict: a dict in which each line contains the pronunciation of one Chinese char\n";
  print STDERR "### oovwordlist: OOV word list\n";
  print STDERR "### oovlex: output OOV lexicon\n";
  exit;
}

use encoding utf8;
my %prons;
open(DICT, $ARGV[0]) || die("Can't open dict ".$ARGV[0]."\n");
foreach (<DICT>) {
  chomp; @A = split(" ", $_); $prons{$A[0]} = $A[1];
}
close DICT;

open(WORDS, $ARGV[1]) || die("Can't open oov word list ".$ARGV[1]."\n");
while (<WORDS>) {
  chomp;
  print $_;
  @A = split("", $_);
  foreach (@A) {
    print " $prons{$_}";
  }
  print "\n";
}
close WORDS;
