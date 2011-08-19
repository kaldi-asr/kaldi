#!/usr/bin/perl -w
# Copyright 2010-2011 Microsoft Corporation

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

# This program returns 0 if the first .scp is a subset of the second
# .scp, and 1 otherwise.  More precisely, the two arguments are text
# files, and only the first field of each is considered; the program
# returns true if the set of first fields of the first are a strict
# subset of the set of first fields of the second.


if(@ARGV < 2 ) {
    die "Usage: is_subset_scp.pl 1.scp 2.scp ";
}


$scp1 = shift @ARGV;
open(S1, "<$scp1") || die "Opening input scp file $scp1";
$scp2 = shift @ARGV;
open(S2, "<$scp2") || die "Opening input scp file $scp2";

while (<S1>) {
    @A = split(" ", $_);
    if (@A == 0) { die "Empty line in first .scp"; }
    $s1{$A[0]} = 1;
}

while (<S2>) {
    @A = split(" ", $_);
    if (@A == 0) { die "Empty line in second .scp"; }
    $s2{$A[0]} = 1;
}

foreach $key ( keys %s1 ) {
    if ( ! defined $s2{$key} ) { 
        print STDERR "is_subset_scp.pl: $scp1 is not a subset of $scp2\n"; 
        exit(-1);
    }
}

exit(0);


