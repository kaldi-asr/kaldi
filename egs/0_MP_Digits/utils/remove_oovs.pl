#!/usr/bin/env perl
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

# This script removes lines that contain these OOVs on either the
# third or fourth fields  of the line.  It is intended to remove arcs
# with OOVs on, from FSTs (probably compiled from ARPAs with OOVs in).

if (  @ARGV < 1 && @ARGV > 2) {
    die "Usage: remove_oovs.pl unk_list.txt [ printed-fst ]\n";
}

$unklist = shift @ARGV;
open(S, "<$unklist") || die "Failed opening unknown-symbol list $unklist\n";
while(<S>){ 
    @A = split(" ", $_);
    @A == 1 || die "Bad line in unknown-symbol list: $_";
    $unk{$A[0]} = 1;
}

$num_removed = 0;
while(<>){ 
    @A = split(" ", $_);
    if(defined $unk{$A[2]} || defined $unk{$A[3]}) {
        $num_removed++;
    } else {
        print;
    }
}
print STDERR "remove_oovs.pl: removed $num_removed lines.\n";

