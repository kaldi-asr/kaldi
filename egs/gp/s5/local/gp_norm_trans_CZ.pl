#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter

# Copyright 2012  Milos Janda

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


# This script normalizes the GlobalPhone Czech transcripts that have been 
# extracted in a format where each line contains an utterance ID followed by
# the transcript, e.g:
# CZ001_2 je tento reklamní slogan pravdivý?
# The normalization is similar to that in 'gp_norm_dict_CZ.pl' script.

my $usage = "Usage: gp_norm_trans_CZ.pl [-u] -i transcript > formatted\
Normalizes transcriptions for GlobalPhone Czech. The input format is \
assumed to be utterance ID followed by transcript on the same line. \
Input is assumed to be ISO-8859-2 encoded, and output is in UTF-8. \
Transcript is lowercased by default, but can be uppercased with the -u option.
\n";

use strict;
use Getopt::Long;
use Unicode::Normalize;
use open ':encoding(iso-8859-2)';
binmode(STDOUT, ":encoding(utf8)");

die "$usage" unless(@ARGV >= 1);
my ($in_trans, $uppercase);
GetOptions ("u" => \$uppercase,    # convert words to uppercase
	    "i=s" => \$in_trans);  # Input transcription

open(T, "<$in_trans") or die "Cannot open transcription file '$in_trans': $!";
while (<T>) {
  s/\r//g;  # Since files may have CRLF line-breaks!
  chomp;
  $_ =~ m:^(\S+)\s+(.+): or die "Bad line: $_";
  my $utt_id = $1;
  my $trans = $2;

  $trans =~ s/ \,(.*?)\'/ $1/g;  # Remove quotation marks.
  # Remove all special characters  
  $trans =~ s/[\;\:\`\<\>\,\.\-\?\!\@\#\$\%\&\(\)\[\]\{\}\"\/\']/ /g;
  # Normalize spaces
  $trans =~ s/^\s*//; $trans =~ s/\s*$//; $trans =~ s/\s+/ /g;

  if (defined($uppercase)) {
    $trans = uc($trans);
  } else {
    $trans = lc($trans);
  }

  print "$utt_id $trans\n";
}
