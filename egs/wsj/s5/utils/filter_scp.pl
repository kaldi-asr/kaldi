#!/usr/bin/perl
# Copyright 2010-2012 Microsoft Corporation
#                     Johns Hopkins University (author: Daniel Povey)

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


# This script takes a list of utterance-ids or any file whose first field
# of each line is an utterance-id, and filters an scp
# file (or any file whose "n-th" field is an utterance id), printing
# out only those lines whose "n-th" field is in id_list. The index of 
# the "n-th" field is zero, by default, but can be changed by using \
# the -f <n> switch

$exclude = 0;
$field = 0;
$shifted = 0;

do {
  $shifted=0;
  if ($ARGV[0] eq "--exclude") {
    $exclude = 1;
    shift @ARGV;
    $shifted=1;
  }
  if ($ARGV[0] eq "-f") {
    $field = $ARGV[1];
    shift @ARGV; shift @ARGV;
    $shifted=1
  }
} while ($shifted);

if(@ARGV < 1 || @ARGV > 2) {
  die "Usage: filter_scp.pl [--exclude] [-f <field-to-filter-on>] id_list [in.scp] > out.scp ";
}


$idlist = shift @ARGV;
open(F, "<$idlist") || die "Could not open id-list file $idlist";
while(<F>) {
  @A = split;
  @A>=1 || die "Invalid id-list file line $_";
  $seen{$A[0]} = 1;
}

while(<>) {
  @A = split;
  @A > 0 || die "Invalid scp file line $_";
  @A >= $field || die "Invalid scp file line $_";

  if((!$exclude && $seen{$A[$field]}) || ($exclude && !defined $seen{$A[$field]})) {
    print $_;
  }
}
