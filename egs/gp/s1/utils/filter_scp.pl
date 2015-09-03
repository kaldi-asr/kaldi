#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter
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


# This script takes a list of utterance-ids or any file whose first field
# of each line is an utterance-id, and filters an scp
# file (or any file whose first field is an utterance id), printing
# out only those lines whose first field is in id_list.

if(@ARGV < 1 || @ARGV > 2) {
    die "Usage: filter_scp.pl id_list [in.scp] > out.scp ";
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
    if($seen{$A[0]}) {
        print $_;
    }
}
