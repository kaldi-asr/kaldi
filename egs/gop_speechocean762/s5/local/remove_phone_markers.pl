#!/usr/bin/env perl
# Copyright 2019  Junbo Zhang
#           2020  Xiaomi Corporation (Author: Junbo Zhang)

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

use strict;
use warnings;

my $Usage = <<EOU;
remove_phone_markers.pl:
This script processes a phone set (i.e. the phones.txt file), remove the stress
markers and the pos-in-word markers, and creates a new phone.txt file and an
old->new phone mapping file, in which each line is: "old-integer-id new-integer-id.

Usage: utils/remove_phone_markers.pl <old-phone-symbols> <new-phone-symbols> <mapping>
 e.g.: utils/remove_phone_markers.pl phones.txt phones-pure.txt phone-to-pure-phone.int
EOU

if (@ARGV < 3) {
  die $Usage;
}

my $old_phone_symbols_filename = shift @ARGV;
my $new_phone_symbols_filename = shift @ARGV;
my $mapping_filename = shift @ARGV;

my %id_of_old_phone;
open(IN, $old_phone_symbols_filename) or die "Can't open $old_phone_symbols_filename";
while (<IN>) {
  chomp;
  my ($phone, $id) = split;
  next if $phone =~ /\#/;
  $id_of_old_phone{$phone} = $id;
}
close IN;

my $new_id = 0;
my %id_of_new_phone;
my %id_old_to_new;
foreach (sort { $id_of_old_phone{$a} <=> $id_of_old_phone{$b} } keys %id_of_old_phone) {
  my $old_phone = $_;
  s/_[BIES]//;
  s/\d//;
  my $new_phone = $_;
  $id_of_new_phone{$new_phone} = $new_id++ if not exists $id_of_new_phone{$new_phone};
  $id_old_to_new{$id_of_old_phone{$old_phone}} = $id_of_new_phone{$new_phone};
}

# Write to file
open(OUT, ">$new_phone_symbols_filename") or die "Can\'t write to $new_phone_symbols_filename";
foreach (sort { $id_of_new_phone{$a} <=> $id_of_new_phone{$b} } keys %id_of_new_phone) {
  print OUT "$_\t$id_of_new_phone{$_}\n";
}
close OUT;

open(OUT, ">$mapping_filename") or die "Can\'t write to $mapping_filename";
foreach (sort { $a <=> $b } keys %id_old_to_new) {
  next if $_ == 0;
  print OUT "$_ $id_old_to_new{$_}\n";
}
close OUT;
