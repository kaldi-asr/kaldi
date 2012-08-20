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


# This script maps from the GlobalPhone-style ASCII encoding of Russian 
# characters to UTF8. This is almost identical to the corresponding script for
# Portuguese and Spanish

use strict;
use Unicode::Normalize;

binmode(STDOUT, ":encoding(utf8)");

my $usage = "Usage: gp_rmn2utf_RU.pl < input > utf8-output\
Maps from the GlobalPhone-style ASCII encoding of Spanish characters to UTF8.\n";

if (defined($ARGV[0])) {
  if ($ARGV[0] =~ m/(-h|--help)/) {
    print STDERR "$usage";
    exit 0;
  } else {
    die "Unknown option '$ARGV[0]'\n$usage";
  }
}

while (<STDIN>) {
  s/~/\x{00D8}/g; # mekky znak - zmekcuje souhlasku pred nim napr t~ => Tb

  s/schTsch/\x{00DD}/g;
  s/SchTsch/\x{00DD}/g;

  s/jscH/\x{00D6}/g;
  s/JscH/\x{00D6}/g;
  s/tscH/\x{00DE}/g;
  s/TscH/\x{00DE}/g;
  s/sch/\x{00DB}/g;
  s/Sch/\x{00DB}/g;
  s/ts/\x{00C3}/g;
  s/tS/\x{00C3}/g;
  s/Ts/\x{00C3}/g;
  s/TS/\x{00C3}/g;

  s/ye/\x{00C5}/g;
  s/yo/\x{00A3}/g; # neni v textu
  s/yu/\x{00C0}/g;
  s/ya/\x{00D1}/g;

  s/Ye/\x{00C5}/g;
  s/Yo/\x{00A3}/g; # neni v textu
  s/Yu/\x{00C0}/g;
  s/Ya/\x{00D1}/g;

  s/i2/\x{00D9}/g;
  s/I2/\x{00D9}/g;
  s/Q/\x{00DF}/g;

  s/a/\x{00C1}/g;
  s/b/\x{00C2}/g;
  s/w/\x{00D7}/g;
  s/g/\x{00C7}/g;
  s/d/\x{00C4}/g;
  s/z/\x{00DA}/g;
  s/i/\x{00C9}/g;
  s/j/\x{00CA}/g;
  s/k/\x{00CB}/g;
  s/l/\x{00CC}/g;
  s/m/\x{00CD}/g;
  s/n/\x{00CE}/g;
  s/o/\x{00CF}/g;
  s/p/\x{00D0}/g;
  s/r/\x{00D2}/g;
  s/s/\x{00D3}/g;
  s/t/\x{00D4}/g;
  s/u/\x{00D5}/g;
  s/f/\x{00C6}/g;
  s/h/\x{00C8}/g;
  s/e/\x{00DC}/g;

  s/A/\x{00C1}/g;
  s/B/\x{00C2}/g;
  s/W/\x{00D7}/g;
  s/G/\x{00C7}/g;
  s/D/\x{00C4}/g;
  s/Z/\x{00DA}/g;
  s/I/\x{00C9}/g;
  s/J/\x{00CA}/g;
  s/K/\x{00CB}/g;
  s/L/\x{00CC}/g;
  s/M/\x{00CD}/g;
  s/N/\x{00CE}/g;
  s/O/\x{00CF}/g;
  s/P/\x{00D0}/g;
  s/R/\x{00D2}/g;
  s/S/\x{00D3}/g;
  s/T/\x{00D4}/g;
  s/U/\x{00D5}/g;
  s/F/\x{00C6}/g;
  s/H/\x{00C8}/g;
  s/E/\x{00DC}/g;
  
  print NFC($_);  # recompose & reorder canonically
}
