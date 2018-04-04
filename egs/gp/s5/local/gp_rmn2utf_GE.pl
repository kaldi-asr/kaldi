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


# This script maps from the GlobalPhone-style ASCII encoding of German 
# characters to UTF8

use strict;
use Unicode::Normalize;

binmode(STDOUT, ":encoding(utf8)");

my $usage = "Usage: gp_rmn2utf_GE.pl < input > utf8-output\
Maps from the GlobalPhone-style ASCII encoding of German characters to UTF8.\n";

if (defined($ARGV[0])) {
  if ($ARGV[0] =~ m/(-h|--help)/) {
    print STDERR "$usage";
    exit 0;
  } else {
    die "Unknown option '$ARGV[0]'\n$usage";
  }
}

while (<STDIN>) {
  s/\~A/\x{00C4}/g;
  s/\~O/\x{00D6}/g;
  s/\~U/\x{00DC}/g;
  
  s/\~a/\x{00E4}/g;
  s/\~o/\x{00F6}/g;
  s/\~u/\x{00FC}/g;
  s/\~s/\x{00DF}/g;

  print NFC($_);  # recompose & reorder canonically
}
