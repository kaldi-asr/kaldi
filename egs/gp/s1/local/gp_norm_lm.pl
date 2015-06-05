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


# This script normalizes the GlobalPhone language models that have been supplied
# with the corpus (at least at Edinbugh). It expects an ARPA-format (unzipped) LM as input.

my $usage = "Usage: gp_format_lm.pl [-a] -i lm > formatted\
Normalizes language models (in ARPA format) for GlobalPhone.\
Options:\
  -a\tTreat acronyms differently (puts - between individual letters)\n";

use strict;
use Getopt::Long;
die "$usage" unless(@ARGV >= 1);
my ($acro, $in_lm, @ngram_counts);
GetOptions ("a" => \$acro,      # put - between letters of acronyms
            "i=s" => \$in_lm);  # Input language model

open(G, "<$in_lm") or die "Cannot open language model file '$in_lm': $!";

while (<G>) { last if /^\\data\\/; }  # Skip till header
print;

# Read counts for various n-gram orders
while (<G>) {
  print;
  next if /^$/;
  last if /^\\1-grams\:/;
  m/^ngram (\d+)\s*\=\s*(\d+)$/ or die "Bad line: $_";
  $ngram_counts[$1] = $2;  # Not 0-indexed!
}

my $nproc;
for my $i (1..$#ngram_counts-1) {  # For all except highest n-gram order
  $nproc = 0;  # Number of n-grams actually found.
  while (<G>) {
    if ($_ =~ /^$/) { print; next; }
    if ($_ =~ /^\\(\d+)\-grams\:/) {
      die "Expecting $ngram_counts[$i] $i-grams; found $nproc!" 
	unless ($nproc == $ngram_counts[$i]);
      print; last; 
    }
    my @fields = split;
    die "Bad line: $_" if ($#fields<$i || $#fields>$i+1);
    for my $f (1..$i) {
      &NormalizeWord($fields[$f]);
    }
    print join(" ", @fields), "\n";
    $nproc += 1;
  }
}

$nproc = 0;
my $i = $#ngram_counts;
while (<G>) {  # Now, process the entries for the highest n-gram order
  if ($_ =~ /^$/) { print; next; }
  if ($_ =~ /^\\end\\/) { 
    die "Expecting $ngram_counts[$i] ${i}-grams; found $nproc!" 
      unless ($nproc == $ngram_counts[$i]);
    print; last; 
  }
  my @fields = split;
  die "Bad line: $_" if ($#fields != $i);
  for my $f (1..$i) {
    &NormalizeWord($fields[$f]);
  }
  print join(" ", @fields), "\n";
  $nproc += 1;
}


sub NormalizeWord {
  my $word = $_[0];
  $word =~ s/\(.*\)//g;  # Pron variants should have same orthography

  # Distinguish acronyms before capitalizing everything, since they have 
  # different pronunciations. This may not be important.
  if (defined($acro)) {
    if ($word =~ /^[A-Z-]+(\-.*)*$/) {
      my @subwords = split('-', $word);
      $word = "";
      for my $i (0..$#subwords) {
	if($subwords[$i] =~ /^[A-Z]{2,}$/) {
	  $subwords[$i] = join('-', split(//, $subwords[$i]));
	}
      }
      $word = join('-', @subwords);
    }
  }
  $word =~ tr/a-z/A-Z/;  # Now, capitalize every word.
  $word =~ s:<S>:<s>:g;
  $word =~ s:</S>:</s>:g;
  $_[0] = $word;
}
