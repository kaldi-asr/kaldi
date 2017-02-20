#!/usr/bin/env perl
#===============================================================================
# Copyright 2017  (Author: Yenda Trmal <jtrmal@gmail.com>)
#
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
#===============================================================================

use strict;
use warnings;
use utf8;

open(my $F, "<", "../src/.version") or do {
  print "$!\n";
  print "The file ../src/.version does not exist\n";
  print "Either you are not running this script from within\n";
  print "the windows/ directory or you have accidently \n";
  print "delete the file\n";
  exit 1;
};

open(my $H, ">", "../src/base/version.h") or do {
  print "$!\n";
  print "Could not write to ../src/base/version.h\n";
  print "Either you are not running this script from within\n";
  print "the windows/ directory or there were some other \n";
  print "issues\n";
  exit 1;
};

my $kaldi_ver=<$F>; chomp $kaldi_ver;
print $H  "#define KALDI_VERSION \"${kaldi_ver}-win\"\n";
close($F);
close($H);
