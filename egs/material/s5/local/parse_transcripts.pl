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

binmode STDIN, "utf8";
binmode STDOUT, "utf8";
binmode STDERR, "utf8";

my $file = $ARGV[0];

open(my $transcript, "<:utf8", $file) or
  die "Cannot open file $file: $!\n";

(my $basename = $file) =~ s/(.*\/)?([^\/]+)/$2/g;

my $sentence = undef;
my $begin_time = undef;
my $end_time = undef;
while(<$transcript>) {
  chomp;
  if (/^\[([0-9.]+)\]$/) {
    $begin_time = $end_time;
    $end_time = $1;
    if ($sentence) {
      print "$basename\t$begin_time\t$end_time\t$sentence\n";
      $sentence = undef;
    }
  } else {
    die "Invalid format of the transcription in $basename\n" if defined($sentence);
    $sentence = $_;
  }
}

die "Invalid format of the transcription in $basename\n" if defined($sentence);

