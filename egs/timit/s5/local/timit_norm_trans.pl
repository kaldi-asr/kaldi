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


# This script normalizes the TIMIT phonetic transcripts that have been 
# extracted in a format where each line contains an utterance ID followed by 
# the transcript, e.g.:
# fcke0_si1111 h# hh ah dx ux w iy dcl d ix f ay n ih q h#

my $usage = "Usage: timit_norm_trans.pl -i transcript -m phone_map -from [60|48] -to [48|39] > normalized\n
Normalizes phonetic transcriptions for TIMIT, by mapping the phones to a 
smaller set defined by the -m option. This script assumes that the mapping is 
done in the \"standard\" fashion, i.e. to 48 or 39 phones.  The input is 
assumed to have 60 phones (+1 for glottal stop, which is deleted), but that can
be changed using the -from option. The input format is assumed to be utterance 
ID followed by transcript on the same line.\n";

use strict;
use Getopt::Long;
die "$usage" unless(@ARGV >= 1);
my ($in_trans, $phone_map, $num_phones_out);
my $num_phones_in = 60;
GetOptions ("i=s" => \$in_trans,          # Input transcription
	    "m=s" => \$phone_map,         # File containing phone mappings
	    "from=i" => \$num_phones_in,  # Input #phones: must be 60 or 48
	    "to=i" => \$num_phones_out ); # Output #phones: must be 48 or 39

die $usage unless(defined($in_trans) && defined($phone_map) && 
		  defined($num_phones_out));
if ($num_phones_in != 60 && $num_phones_in != 48) {
  die "Can only used 60 or 48 for -from (used $num_phones_in)."
}
if ($num_phones_out != 48 && $num_phones_out != 39) {
  die "Can only used 48 or 39 for -to (used $num_phones_out)."
}
unless ($num_phones_out < $num_phones_in) {
  die "Argument to -from ($num_phones_in) must be greater than that to -to ($num_phones_out)."
}


open(M, "<$phone_map") or die "Cannot open mappings file '$phone_map': $!";
my (%phonemap, %seen_phones);
my $num_seen_phones = 0;
while (<M>) {
  chomp;
  next if ($_ =~ /^q\s*.*$/); # Ignore glottal stops.
  m:^(\S+)\s+(\S+)\s+(\S+)$: or die "Bad line: $_";
  my $mapped_from = ($num_phones_in == 60)? $1 : $2;
  my $mapped_to = ($num_phones_out == 48)? $2 : $3;
  if (!defined($seen_phones{$mapped_to})) {
    $seen_phones{$mapped_to} = 1;
    $num_seen_phones += 1;
  }
  $phonemap{$mapped_from} = $mapped_to;
}
if ($num_seen_phones != $num_phones_out) {
  die "Trying to map to $num_phones_out phones, but seen only $num_seen_phones";
}

open(T, "<$in_trans") or die "Cannot open transcription file '$in_trans': $!";
while (<T>) {
  chomp;
  $_ =~ m:^(\S+)\s+(.+): or die "Bad line: $_";
  my $utt_id = $1;
  my $trans = $2;

  $trans =~ s/q//g;  # Remove glottal stops.
  $trans =~ s/^\s*//; $trans =~ s/\s*$//;  # Normalize spaces

  print $utt_id;
  for my $phone (split(/\s+/, $trans)) {
    print " $phonemap{$phone}"
  }
  print "\n";
}
