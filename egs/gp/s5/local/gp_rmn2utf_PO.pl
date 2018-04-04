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


# This script maps from the GlobalPhone-style ASCII encoding of Portuguese 
# characters to UTF8

use strict;
use Unicode::Normalize;

binmode(STDOUT, ":encoding(utf8)");

my $usage = "Usage: gp_rmn2utf_PO.pl < input > utf8-output\
Maps from the GlobalPhone-style ASCII encoding of Portuguese characters to UTF8.\n";

if (defined($ARGV[0])) {
  if ($ARGV[0] =~ m/(-h|--help)/) {
    print STDERR "$usage";
    exit 0;
  } else {
    die "Unknown option '$ARGV[0]'\n$usage";
  }
}

while (<STDIN>) {
  s/A\:/\x{00C0}/g;
  s/A\+/\x{00C1}/g;
  s/A\^/\x{00C2}/g;
  s/A\~/\x{00C3}/g;
  s/C\:/\x{00C7}/g;
  s/E\+/\x{00C9}/g;
  s/E\^/\x{00CA}/g;
  s/I\+/\x{00CD}/g;
  s/N\~/\x{00D1}/g;
  s/O\+/\x{00D3}/g;
  s/O\^/\x{00D4}/g;
  s/O\~/\x{00D5}/g;
  s/U\+/\x{00DA}/g;
  s/U\^/\x{00DC}/g;

  s/a\:/\x{00E0}/g;
  s/a\+/\x{00E1}/g;
  s/a\^/\x{00E2}/g;
  s/a\~/\x{00E3}/g;
  s/c\:/\x{00E7}/g;
  s/e\+/\x{00E9}/g;
  s/e\^/\x{00EA}/g;
  s/i\+/\x{00ED}/g;
  s/n\~/\x{00F1}/g;
  s/o\+/\x{00F3}/g;
  s/o\^/\x{00F4}/g;
  s/o\~/\x{00F5}/g;
  s/u\+/\x{00FA}/g;
  s/u\^/\x{00FC}/g;  
  
  print NFC($_);  # recompose & reorder canonically
}
