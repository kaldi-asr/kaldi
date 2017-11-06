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

# replacement of the smart-match operator (apparently not supported anymore)
sub is_elem {
  my $word = shift;
  my $array = shift;
  foreach my $other_word (@{$array}) {
    return 1 if $word eq $other_word;
  }
  return 0;
}

my $unk = "<unk>";
my $noise = "<noise>";
my $spnoise = "<spnoise>";
my $sil = "<sil>";

my @ignore_events = ("<female-to-male>", "<male-to-female>");
#as per the BABEL docs, ~ means truncation of the word/utterance
my @ignore_utt_events = ("<overlap>", "<dtmf>", "<foreign>", "~");
my @sil_events = ("<no-speech>");
my @noise_events = ("<sta>", "<ring>", "<int>" );
my @spnoise_events = ("<breath>", "<cough>", "<hes>", "<laugh>", "<click>", "<lipsmack>");



UTT: while(<>) {
  chomp;
  my @line = split " ", $_;
  my $file = shift @line;
  my $begin = shift @line;
  my $end = shift @line;

  next if (@line == 1) and ($line[0] eq "<no-speech>");
  next if (@line == 1) and ($line[0] =~ "<.*>"); #skip the utterance if all
                                                 #it contains is a non-speech event

  my @out_line;
  foreach my $word (@line) {
    if ($word =~ /.*-$/) {
      push @out_line, $unk;
    } elsif ($word =~ /^-.*/) {
      push @out_line, $unk;
    } elsif ($word =~ /^\*.*\*$/) {
      push @out_line, $unk;
    } elsif ($word eq "(())") {
      push @out_line, $unk;
    } elsif (is_elem $word, \@ignore_events) {
      next;
    } elsif (is_elem $word, \@ignore_utt_events) {
      next UTT;
    } elsif (is_elem $word, \@sil_events) {
      push @out_line, $sil;
    } elsif (is_elem $word, \@noise_events) {
      push @out_line, $noise;
    } elsif (is_elem $word, \@spnoise_events) {
      push @out_line, $spnoise;
    } else {
      push @out_line, $word;
    }
  }
  print "$file\t$begin\t$end\t" . join(" ", @out_line) . "\n" if @out_line;

}


