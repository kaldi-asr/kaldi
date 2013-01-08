#!/usr/bin/perl
# Copyright 2012 Navdeep Jaitly. 

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


# makes lexicon FST (no pron-probs involved).

if(@ARGV != 2) {
    die "Usage: make_lexicon_fst.pl lexicon.txt silphone "
}

$lexfn = shift @ARGV;
$silphone = shift @ARGV;

open(L, "<$lexfn") || die "Error opening lexicon $lexfn";

$loopstate=1 ;
$silstate=2 ;
$sil_cost=-log(0.5);
print "0\t$loopstate\t<eps>\t<eps>\n";
print "$loopstate\t$silstate\t$silphone\t<eps>\t$sil_cost\n";
print "$silstate\t$loopstate\t<eps>\t<eps>\n";
while(<L>) {
	 @A = split(" ", $_);
	 $w = shift @A;
    (@A == 1) || die "Incorrect line in lexicon.txt:  $_";
	 $p = shift @A;
	 print "$loopstate\t$loopstate\t$p\t$w\n";
}
print "$loopstate\t0\n"; # final-cost.
