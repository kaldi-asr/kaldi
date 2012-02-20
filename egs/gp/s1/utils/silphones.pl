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


# creates integer lists of silence and non-silence phones in files,
# e.g. silphones.csl="1:2:3 \n"
# and nonsilphones.csl="4:5:6:7:...:24\n";

if(@ARGV != 4) {
    die "Usage: silphones.pl phones.txt \"sil1 sil2 sil3\" silphones.csl nonsilphones.csl";
}

($symtab, $sillist, $silphones, $nonsilphones) = @ARGV;
open(S,"<$symtab") || die "Opening symbol table $symtab";


foreach $s (split(" ", $sillist)) {
    $issil{$s} = 1;
}

@sil = ();
@nonsil = ();
while(<S>){
    @A = split(" ", $_);
    @A == 2 || die "Bad line $_ in phone-symbol-table file $symtab";
    ($sym, $int) = @A;
    if($int != 0) {
        if($issil{$sym}) { push @sil, $int; $seensil{$sym}=1; }
        else { push @nonsil, $int; }
    }
}

foreach $k(keys %issil) {
    if(!$seensil{$k}) { die "No such silence phone $k"; }
}
open(F, ">$silphones") || die "opening silphones file $silphones";
open(G, ">$nonsilphones") || die "opening nonsilphones file $nonsilphones";
print F join(":", @sil) . "\n";
print G join(":", @nonsil) . "\n";
close(F);
close(G);
if(@sil == 0) { print STDERR "Warning: silphones.pl no silence phones.\n" }
if(@nonsil == 0) { print STDERR "Warning: silphones.pl no non-silence phones.\n" }

