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

# 'phonesets_mono' contains sets of phones that are shared when building the 
# monophone system and when asking questions based on an automatic clustering 
# of phones, for the triphone system.  
# 'roots' contain the information about which phones share a common root in 
# the phonetic decision tree and which have distinct pdfs. It also states 
# whether the tree-building should split the roots or not.

my $usage = "Usage: timit_make_questions.pl -i phones -m phoneset_mono -r roots\
Creates sharerd phonesets for monophone and context-dependent training.\
Required arguments:\
  -i\tInput list of phones (can contain stress/position markers)\
  -m\tOutput shared phoneset for use in monophone training\
  -r\tOutput sharing and splitting info for context-dependent training\n";

use strict;
use Getopt::Long;
my ($in_phones, $mono, $roots, %phoneset);
GetOptions ("i=s" => \$in_phones,  # Input list of phones
            "m=s" => \$mono,       # Shared phone-set for monophone system
	    "r=s" => \$roots );    # roots file for context-dependent systems

die "$usage" unless(defined($in_phones) && defined($mono) && defined($roots));

open(P, "<$in_phones") or die "Cannot read from file '$in_phones': $!";
open(MONO, ">$mono") or die "Cannot write to file '$mono': $!";
open(ROOTS, ">$roots") or die "Cannot write to file '$roots': $!";

while (<P>) {
  next if m/eps|sil|vcl|cl|epi/;
  chomp;
  m/^(\S+)(_.)?\s+\S+$/ or die "Bad line: $_\n";
  my $full_phone = defined($2)? $1.$2 : $1;
  push @{$phoneset{$1}}, $full_phone;
}

print MONO "cl epi sil vcl\n";
print ROOTS "not-shared not-split cl epi sil vcl\n";
foreach my $p (sort keys %phoneset) {
  print MONO join(" ", @{$phoneset{$p}}), "\n";
  print ROOTS "shared split ", join(" ", @{$phoneset{$p}}), "\n";
}
