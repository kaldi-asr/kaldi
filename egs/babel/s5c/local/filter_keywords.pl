#!/usr/bin/perl

use strict;
use warnings;
use Encode;

my $Usage = <<EOU;
Usage:    filter_keywords.pl <dictin> <queryin|-> <queryout|->

EOU

if(@ARGV != 3) {
  die $Usage;
}

# Get parameters
my $dictin = shift @ARGV;
my $filein = shift @ARGV;
my $fileout = shift @ARGV;

# Open dictionary
if (!open(D, "<$dictin")) {print "Fail to open dictionary: $dictin\n"; exit 1;}

# Get input source
my $source = "";
if ($filein eq "-") {
  $source = "STDIN";
} else {
  if (!open(I, "<$filein")) {print "Fail to open input file: $filein\n"; exit 1;}
  $source = "I";
}

# Open output fst list
my $sourceout = "";
if ($fileout ne "-") {
  if (!open(O, ">$fileout")) {print "Fail to open output file: $fileout\n"; exit 1;}
  $sourceout = "O";
}

# Read in the dictionary
my %dict = ();
while (<D>) {
  chomp;
  my @col = split(" ", $_);
  my $word = shift @col;
  my $original_w = $word;
  $word =~ tr/a-z/A-Z/;
  $dict{$word} = $original_w;
}

# Process the queries
my $word;
while (<$source>) {
  chomp;
  my @col = split(" ", $_);
  foreach $word (@col) {
    if (defined($dict{$word})) {
      eval "print $sourceout \"$dict{$word} \"";
    } else {
      eval "print $sourceout \"$word \"";
    }
  }
  eval "print $sourceout \"\n\"";
}

close(D);
if ($filein  ne "-") {close(I);}
if ($fileout ne "-") {close(O);}
