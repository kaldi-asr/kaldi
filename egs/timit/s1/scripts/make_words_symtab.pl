#!/usr/bin/perl
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

# make_words_symtab.pl < lexicon.txt > words.txt


while(<>) {
    @A = split(" ", $_);
        $W{$A[0]} = 1;
}

print "<eps>\t0\n";
$n = 1;
foreach $w (sort keys %W) {
    if($w ne "<eps>") {
        print "$w\t$n\n";
        $n++;
    }
}

print "!SIL\t$n\n";

