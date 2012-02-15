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


# This script builds a lower-order LM.

my $usage = "Usage: subset_lm.pl -i input[.gz] -n order -o output[.gz]
 Builds a lower-oder LM from the input (can read & write gzipped files).\n";

use strict;
use Getopt::Long;
my ($in_lm, $order, $out_lm, @ngram_counts);
GetOptions ("i=s" => \$in_lm,   # Input language model file
	    "n=i" => \$order,   # Max n-gram order of the output
            "o=s" => \$out_lm); # Output language model file

die "$usage" unless (defined($in_lm) && defined($order) && defined($out_lm));
die "Output order must be positive." unless ($order > 0);

my $cmd = ($in_lm =~ /\.gz$/)? "gzip -dc $in_lm |" : "<$in_lm";
open(IN, "$cmd") or die "Cannot read from language model file '$in_lm': $!";
$cmd = ($out_lm =~ /\.gz$/)? "| gzip -c > $out_lm" : ">$out_lm";
open(OUT, "$cmd") or die "Cannot write to language model file '$out_lm': $!";

while (<IN>) { last if /^\\data\\/; }  # Skip till header

# Read counts for various n-gram orders
while (<IN>) {
  next if /^$/;
  last if /^\\1-grams\:/;
  m/^ngram (\d+)\s*\=\s*(\d+)$/ or die "Bad line: $_";
  $ngram_counts[$1] = $2;  # Not 0-indexed!
}

# Handle degenerate case.
if ($#ngram_counts <= $order) {
  warn "Input LM order ($#ngram_counts) <= required LM order ($order): input and output will be identical.";
  seek(IN, 0, 0);
  while (<IN>) { print OUT; }  # Don't have to worry about gz 
  close IN; close OUT;
  exit 0;
} else {  # Print the header (with lower n-gram order)
  print OUT "\\data\\\n";
  for my $i (1..$order) {
    print OUT "ngram $i=$ngram_counts[$i]\n";
  }
  print OUT "\n";  
}

my $nproc;
for my $i (1..$order) {
  $nproc = 0;  # Number of n-grams actually found.
  print OUT "\n\\${i}-grams:\n";
  while (<IN>) {
    next if ($_ =~ /^$/);
    if ($_ =~ /^\\(\d+)\-grams\:/) {
      die "Expecting $ngram_counts[$i] $i-grams; found $nproc!" 
	unless ($nproc == $ngram_counts[$i]);
      last; 
    }
    my @fields = split;
    die "Bad line: $_" if ($#fields<$i || $#fields>$i+1);
    # For the highest order of n-gram, remove back-off weight, if present.
    pop(@fields) if ($i==$order && $#fields==$i+1);
    print OUT join(" ", @fields), "\n";
    $nproc += 1;
  }
}

print OUT "\\end\\\n";
close IN;
close OUT;
