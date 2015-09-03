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


if (  @ARGV < 1 && @ARGV > 2) {
    die "Usage: find_arpa_oovs.pl words.txt [lm.arpa]\n";
    # This program finds words in the arpa file that are not symbols
    # in the OpenFst-format symbol table words.txt.  It prints them
    # on the standard output, one per line.
}

$symtab = shift @ARGV;
open(S, "<$symtab") || die "Failed opening symbol table file $symtab\n";
while(<S>){ 
    @A = split(" ", $_);
    @A == 2 || die "Bad line in symbol table file: $_";
    $seen{$A[0]} = 1;
}

$curgram=0;
while(<>) { # Find the \data\ marker.
    if(m:^\\data\\$:) { last; }
}
while(<>) {
    if(m/^\\(\d+)\-grams:\s*$/) {
        $curgram = $1;
        if($curgram > 1) {
            last; # This is an optimization as we can get the vocab from the 1-grams
        }
    } elsif($curgram > 0) {
        @A = split(" ", $_);
        if(@A > 1) {
            shift @A;
            for($n=0;$n<$curgram;$n++) {
                $word = $A[$n];
                if(!defined $word) { print STDERR "Unusual line $_ (line $.) in arpa file.\n"; }
                $in_arpa{$word} = 1;
            }
        } else {
            if(@A > 0 && $A[0] !~ m:\\end\\:) {
                print STDERR "Unusual line $_ (line $.) in arpa file\n";
            }
        }
    }
}

foreach $w (keys %in_arpa) {
    if(!defined $seen{$w} && $w ne "<s>" && $w ne "</s>") {
        print "$w\n";
    }
}
