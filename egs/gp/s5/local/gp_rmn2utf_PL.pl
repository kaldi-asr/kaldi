#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter

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


# This script maps from the GlobalPhone-style ASCII encoding of Polish 
# characters to UTF8

use strict;
use Unicode::Normalize;

binmode(STDOUT, ":encoding(utf8)");

my $usage = "Usage: gp_rmn2utf_PL.pl < input > utf8-output\
Maps from the GlobalPhone-style ASCII encoding of Polish characters to UTF8.\n";

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
  s/A\~/\x{0104}/g; # A WITH OGONEK
  s/E\~/\x{0118}/g; # E WITH OGONEK

  s/Z0/\x{017B}/g; # Z WITH DOT ABOVE

  s/C1/\x{0106}/g; # LETTERS WITH ACUTE
  s/L1/\x{0141}/g;
  s/N1/\x{0143}/g;
  s/O1/\x{00D3}/g;
  s/S1/\x{015A}/g;
  s/Z1/\x{0179}/g;

  s/O2/\x{00D6}/g; # O WITH DIAERESIS (German umlaut)
  s/U2/\x{00DC}/g; # U WITH DIAERESIS (German umlaut)
  s/C2/\x{00C7}/g; # C WITH CEDILLA (from French)


  # SMALL LETTERS
  s/a\~/\x{0105}/g;
  s/e\~/\x{0119}/g;

  s/z0/\x{017C}/g;

  s/c1/\x{0107}/g;
  s/l1/\x{0142}/g;
  s/n1/\x{0144}/g;
  s/o1/\x{00F3}/g;
  s/s1/\x{015B}/g;
  s/z1/\x{017A}/g;

  s/o2/\x{00F6}/g;
  s/u2/\x{00FC}/g;
  s/c2/\x{00E7}/g;
  
  print NFC($_);  # recompose & reorder canonically
}
