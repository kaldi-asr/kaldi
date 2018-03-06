#!/usr/bin/env perl

# Copyright 2016 FAU Erlangen (Author: Axel Horndasch)
# Apache 2.0.
#
# Concept: Dan Povey

use strict;
use warnings;
use Getopt::Long;

my $Usage = <<EOU;
Usage:  validate_disambig_sym_file.pl [options] disambig_syms.txt

This scripts checks if the entries of a file containing disambiguation symbols
(word or phone level) are all valid. To be valid the symbols
- must start with the hash mark '#',
- must not contain any whitespace,
- must not be equal to '#-1' (disallowed because it is used internally in some
  FST stuff).

In case the option '--allow-numeric' is used with 'false', the symbols must
also be non-numeric (to avoid overlap with the automatically generated symbols).

Allowed options:
  --allow-numeric (true|false) : Default true. If false, disallow numeric
                                 disambiguation symbols like #0, #1 and so on.

EOU

# Command line options
my $allow_numeric = "true";

# Get the optional command line options
GetOptions(
    "allow-numeric=s" => \$allow_numeric,
    ) or die ($Usage);

if (@ARGV != 1) {
  die($Usage);
}

my $disambig_sym_file = shift @ARGV;

print "$0: Checking validity of file \"$disambig_sym_file\" ...\n";
if (-z $disambig_sym_file) {
  print "$0: The file \"$disambig_sym_file\" is empty or does not exist, exiting ...\n"; exit 1;
}

if (not open(SYMS, "<$disambig_sym_file")) {
  print "$0: Could not open file \"$disambig_sym_file\", exiting ...\n"; exit 1;
}

# Go through the file containing disambiguation symbols line by line
while (<SYMS>) {
  chomp;
  my $symbol = $_;

  if ($symbol =~ /^#(.*)$/) {
    my $sympart = $1;
    if ($sympart eq "") {
      print "$0: Only \"$symbol\" is not allowed as a disambiguation symbol, exiting ...\n"; exit 1;
    }
    if ($sympart =~/\s+/) {
      print "$0: The disambiguation symbol \"$symbol\" contains whitespace, exiting ...\n"; exit 1;
    }
    if ($sympart eq "-1") {
      print "$0: The disambiguation symbol \"$symbol\" is not allowed, exiting ...\n"; exit 1;
    }
    if ($allow_numeric eq "false" &&
	$sympart =~/^[0-9]+$/) {
      print "$0: Since \"$symbol\" is supposed to be an extra disambiguation symbol, it must not be numeric, exiting ...\n"; exit 1;
    }
  } else {
    print "$0: The disambiguation symbol \"$symbol\" does not start with a '#', exiting ...\n"; exit 1;
  }
}

print "--> SUCCESS [validating disambiguation symbol file \"$disambig_sym_file\"]\n";
exit 0;

