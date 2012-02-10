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


# This script takes as input any two files in an "scp-like" format that
# has e.g. utterance id's as the first token on each line; and it outputs
# something that will normally be identical to the first file, except that
# if any utterance-id was present in the first input but not the second, it
# will choose the second.  Basically this implements a form of backoff.

if(@ARGV != 2) {
    die "Usage: backoff_scp.pl in1.scp in2.scp > out.scp ";
}

($f1, $f2) = @ARGV;

open(O, "|sort"); # Sort and put into standard out.

open(F1, "<$f1") || die "Could not open input $f1";
while(<F1>) {
    @A = split;
    @A>=1 || die "Invalid id-list file line $_";
    $seen{$A[0]} = 1;
    print O;
}

open(F2, "<$f2") || die "Could not open input $f2";
while(<F2>) {
    @A = split;
    @A > 0 || die "Invalid scp file line $_";
    if(! $seen{$A[0]}) {
        print O;
    }
}

