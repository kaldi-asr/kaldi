#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter
use strict;

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

# This script reads a list of Romanized GlobalPhone transcript files from the 
# standard input (e.g. German/rmn/GE008.rmn). 
# Following the conventions of the corpus, the basename of the transcript file 
# is assumed to be the ID of the speaker (i.e. GE008 in this case). 
# The transcript files are assumed to have the following format:
#   ; 10:
#   man mag es drehen und wenden wie man will
# where the number is the utterance ID. The script prints the utterance ID 
# followed by the transcript, e.g.:
# GE008_10 man mag es drehen und wenden wie man will

while(<STDIN>) {
  chomp;
  $_ =~ m:\S+/(\S+).rmn: || die "Bad line in transcription file list: $_";
  my $spk = $1;
  open(F, "<$_") || die "Error opening transcription file $_\n";
  while(<F>) {
    s/\r//g;  # Since the transcriptions are in DOS format!
    chomp;
    next unless($_ =~ /^;\s*(\d+)\:/);
    my $utt = $1;
    $_ = <F>;
    die "Unexpected line: $_" if($_ =~ /^;/);
    if ($_ =~ /^\s*$/) {
      print STDERR "Empty transcript found for utterance '${spk}_${utt}.\n";
    } else {
      print "${spk}_${utt}\t$_" unless($_ =~ /^$/);
    }
  }
}
