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


$ignore_oov = 0;
$ignore_first_field = 0;
for($x = 0; $x < 3; $x++) {
    # Note: it will just print OOVS unmodified if you specify --ignore-oov.
    # Else will complain and put nothing out.
    if($ARGV[0] eq "--ignore-oov") { $ignore_oov = 1; shift @ARGV; } 
    if($ARGV[0] eq "--ignore-first-field") { $ignore_first_field = 1; shift @ARGV; }
    if($ARGV[0] eq "--map-oov") { shift @ARGV; $map_oov = shift @ARGV; }
}

$symtab = shift @ARGV;
if(!defined $symtab) {
    die "Usage: sym2int.pl symtab [input transcriptions] > output transcriptions\n";
}
open(F, "<$symtab") || die "Error opening symbol table file $symtab";
while(<F>) {
    @A = split(" ", $_);
    @A == 2 || die "bad line in symbol table file: $_";
    $sym2int{$A[0]} = $A[1] + 0;
}

$num_warning = 0;
$max_warning = 20;
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
    @B = ();
    foreach $a (@A) {
        $i = $sym2int{$a};
        if(!defined ($i)) {
            if (defined $map_oov) {
                if (!defined $sym2int{$map_oov}) {
                    die "sym2int.pl: invalid map-oov option $map_oov (symbol not defined in $symtab)";
                }
                if ($num_warning++ < $max_warning) {
                    print STDERR "sym2int.pl: replacing $a with $map_oov\n";
                    if ($num_warning == $max_warning) {
                        print STDERR "sym2int.pl: not warning for OOVs any more times\n";
                    }
                }
                $i = $sym2int{$map_oov};
            } elsif($ignore_oov) {
                $i = $a; # just print them out unmodified..
            } else {
                die "sym2int.pl: undefined symbol $a\n";
            }
        }
        push @B, $i;
    }
    print join(" ", @B);
    print "\n";
}

if($error) { exit(1); }
else { exit(0); }



