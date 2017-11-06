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

my $output = $ARGV[0];
open(my $utt2spk, ">:utf8", "$output/utt2spk") or
  die "Cannot open $output/utt2spk: $!\n";
open(my $text, ">:utf8", "$output/text") or
  die "Cannot open $output/text: $!\n";
open(my $segments, ">:utf8", "$output/segments") or
  die "Cannot open $output/segments: $!\n";
open(my $wav, ">:utf8", "$output/wav2file") or
  die "Cannot open $output/wav2file: $!\n";

my %text2id;
while(<STDIN>) {
  chomp;
  my @line = split (" ", $_, 4);
  my $name = shift @line;
  my $begin =  shift @line;
  my $end = shift @line;
  my $words = shift @line;
  my $name_raw = $name;

  my $begin_text = sprintf("%07d", $begin * 1000);
  my $end_text = sprintf("%07d", $end * 1000);

  # name looks like this:
  #   MATERIAL_BASE-1A-BUILD_10002_20131130_011225_inLine.txt
  # Please note that the naming pattern must match
  # the pattern in audio2wav_scp.pl
  $name =~ s/inLine.*/0/g;
  $name =~ s/outLine.*/1/g;
  $name =~ s/_BASE//g;
  $name =~ s/-BUILD//g;

  my $utt_name = join("_", $name, $begin_text, $end_text);
  print $segments "$utt_name $name $begin $end\n";
  print $utt2spk  "$utt_name $name\n";
  print $text "$utt_name $words\n";
  if (defined $text2id{$name}) {
    die "" if $text2id{$name} ne $name_raw;
  } else {
    print $wav "$name $name_raw\n";
    $text2id{$name} = $name_raw;
  }
}
