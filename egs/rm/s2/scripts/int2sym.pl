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


$ignore_noninteger = 0;
$ignore_first_field = 0;
for($x = 0; $x < 2; $x++) {
    if($ARGV[0] eq "--ignore-noninteger") { $ignore_oov = 1; shift @ARGV; }
    if($ARGV[0] eq "--ignore-first-field") { $ignore_first_field = 1; shift @ARGV; }
}

$symtab = shift @ARGV;
if(!defined $symtab) {
    die "Usage: sym2int.pl symtab [input transcriptions] > output transcriptions\n";
}
open(F, "<$symtab") || die "Error opening symbol table file $symtab";
while(<F>) {
    @A = split(" ", $_);
    @A == 2 || die "bad line in symbol table file: $_";
    $int2sym{$A[1]} = $A[0];
}

$error = 0;
while(<>) {
    @A = split(" ", $_);
    if(@A == 0) {
        die "Empty line in transcriptions input.";
    }
    if($ignore_first_field) {
        $key = shift @A;
        print $key . " ";
    }
    foreach $a (@A) {
        if($a !~  m:^\d+$:) { # not all digits..
            if($ignore_noninteger) {
                print $a . " ";
                next;
            } else {
                if($a eq $A[0]) {
                    die "int2sym.pl: found noninteger token $a (try --ignore-first-field)\n";
                } else {
                    die "int2sym.pl: found noninteger token $a (try --ignore-noninteger if valid input)\n";
                }
            }
        }
        $s = $int2sym{$a};
        if(!defined ($s)) {
            die "int2sym.pl: integer $a not in symbol table $symtab.";
        }
        print $s . " ";
    }
    print "\n";
}



