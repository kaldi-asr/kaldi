#!/usr/bin/perl -w

# Copyright 2012  Arnab Ghoshal

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


# This script maps from the GlobalPhone-style ASCII encoding of French 
# characters to UTF8

use strict;
use Unicode::Normalize;

binmode(STDOUT, ":encoding(utf8)");

my $usage = "Usage: gp_rmn2utf_FR.pl < input > utf8-output\
Maps from the GlobalPhone-style ASCII encoding of French characters to UTF8.\n";

if (defined($ARGV[0])) {
  if ($ARGV[0] =~ m/(-h|--help)/) {
    print STDERR "$usage";
    exit 0;
  } else {
    die "Unknown option '$ARGV[0]'\n$usage";
  }
}

while (<STDIN>) {
  # CAPITAL LETTERS
  s/A\`/\x{00C0}/g; # LETTERS WITH GRAVE
  s/E\`/\x{00C8}/g; 
  s/U\`/\x{00D9}/g;

  s/A\^/\x{00C2}/g; # LETTERS WITH CIRCUMFLEX
  s/E\^/\x{00CA}/g;
  s/I\^/\x{00CE}/g;
  s/O\^/\x{00D4}/g;
  s/U\^/\x{00DB}/g;

  s/E\:/\x{00C8}/g; # LETTERS WITH DIAERESIS
  s/I\:/\x{00CF}/g;
  s/U\:/\x{00DC}/g;
  s/Y\:/\x{0178}/g;

  s/E\+/\x{00C9}/g; # E WITH ACUTE
  s/C\~/\x{00C7}/g; # C WITH CEDILLA

  # SMALL LETTERS
  s/a\`/\x{00E0}/g; # LETTERS WITH GRAVE
  s/e\`/\x{00E8}/g; 
  s/u\`/\x{00F9}/g;

  s/a\^/\x{00E2}/g; # LETTERS WITH CIRCUMFLEX
  s/e\^/\x{00EA}/g;
  s/i\^/\x{00EE}/g;
  s/o\^/\x{00F4}/g;
  s/u\^/\x{00FB}/g;

  s/e\:/\x{00E8}/g; # LETTERS WITH DIAERESIS
  s/i\:/\x{00EF}/g; 
  s/u\:/\x{00FC}/g;
  s/y\:/\x{00FF}/g;

  s/e\+/\x{00E9}/g; # E WITH ACUTE
  s/c\~/\x{00E7}/g; # C WITH CEDILLA

  print NFC($_);  # recompose & reorder canonically
}
