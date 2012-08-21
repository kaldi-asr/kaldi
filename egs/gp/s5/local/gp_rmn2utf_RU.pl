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
  s/~/\x{044C}/g; # Cyrillic Soft Sign - soften consonant before that, e.g. t~ => Tb

  s/schTsch/\x{0449}/g;
  s/SchTsch/\x{0429}/g;

  s/jscH/\x{0436}/g;
  s/JscH/\x{0416}/g;

  s/tscH/\x{0447}/g;
  s/TscH/\x{0427}/g;

  s/sch/\x{0448}/g;
  s/Sch/\x{0428}/g;

  s/ts/\x{0446}/g;
  s/tS/\x{0446}/g;

  s/Ts/\x{0426}/g;
  s/TS/\x{0426}/g;

  s/ye/\x{0435}/g;
  s/yo/\x{0451}/g; # non in text
  s/yu/\x{044E}/g;
  s/ya/\x{044F}/g;

  s/Ye/\x{0415}/g;
  s/Yo/\x{0401}/g; # non in text
  s/Yu/\x{042E}/g;
  s/Ya/\x{042F}/g;

  s/i2/\x{044B}/g;
  s/I2/\x{042B}/g;

  s/Q/\x{044A}/g;
  s/q/\x{042A}/g; # non in text

  s/a/\x{0430}/g;
  s/b/\x{0431}/g;
  s/w/\x{0432}/g;
  s/g/\x{0433}/g;
  s/d/\x{0434}/g;
  s/z/\x{0437}/g;
  s/i/\x{0438}/g;
  s/j/\x{0439}/g;
  s/k/\x{043A}/g;
  s/l/\x{043B}/g;
  s/m/\x{043C}/g;
  s/n/\x{043D}/g;
  s/o/\x{043E}/g;
  s/p/\x{043F}/g;
  s/r/\x{0440}/g;
  s/s/\x{0441}/g;
  s/t/\x{0442}/g;
  s/u/\x{0443}/g;
  s/f/\x{0444}/g;
  s/h/\x{0445}/g;
  s/e/\x{044D}/g;

  s/A/\x{0410}/g;
  s/B/\x{0411}/g;
  s/W/\x{0412}/g;
  s/G/\x{0413}/g;
  s/D/\x{0414}/g;
  s/Z/\x{0417}/g;
  s/I/\x{0418}/g;
  s/J/\x{0419}/g;
  s/K/\x{041A}/g;
  s/L/\x{041B}/g;
  s/M/\x{041C}/g;
  s/N/\x{041D}/g;
  s/O/\x{041E}/g;
  s/P/\x{041F}/g;
  s/R/\x{0420}/g;
  s/S/\x{0421}/g;
  s/T/\x{0422}/g;
  s/U/\x{0423}/g;
  s/F/\x{0424}/g;
  s/H/\x{0425}/g;
  s/E/\x{042D}/g;

  
  print NFC($_);  # recompose & reorder canonically
}
