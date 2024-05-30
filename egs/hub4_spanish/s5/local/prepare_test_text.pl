#!/usr/bin/env perl
#===============================================================================
# Copyright (c) 2017  Johns Hopkins University
#                        (Author: Jan "Yenda" Trmal <jtrmal@gmail.com>)
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

# the main difference between this script and the prepare_train_data.pl is
# that in this script we do not remove nor map the word fragments to <unk>


use strict;
use warnings;
use utf8;

use List::Util qw(max);

binmode(STDOUT, ":utf8");
binmode(STDERR, ":utf8");

sub  trim { my $s = shift; $s =~ s/^\s+|\s+$//g; return $s };

if (@ARGV != 2) {
  print STDERR "Incorrect number of parameters: $ARGV\n";
  print STDERR "Script expects two parameters, symbol of unk and the source text\n";
  die "Example: $0 \"<unk>\" data/eval/text\n";
}

my $unk = $ARGV[0];
open(my $text, "<:encoding(utf-8)", $ARGV[1])
  or die "Cannot open file $ARGV[1]: $!";
while(<$text>) {
  chomp;
  next if /\[\[NS\]\]/;
  (my $utt, my $line) = split / /, $_, 2;
  $line =~ s/<\/?unclear>//g;
  $line =~ s/<.?foreign( [^ >]+)*>//g;
  $line =~ s/{/</g;
  $line =~ s/}/>/g;
  $line =~ s/[,.?^+*]//g;
  $line =~ s/%([^ ]+)/"<" . lc($1) . ">"/ge;

  $line = trim $line;
  next unless $line;
  $line = " $line ";

  $line =~ s/^ +//g;
  $line =~ s/ +$//g;
  print "$utt $line\n";
}
close($text);

