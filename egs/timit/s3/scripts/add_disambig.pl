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


# Adds some specified number of disambig symbols to a symbol table.
# Adds these as #1, #2, etc.
# If the --include-zero option is specified, includes an extra one
# #0.
if(!(@ARGV == 2 || (@ARGV ==3 && $ARGV[0] eq "--include-zero"))) {
    die "Usage: add_disambig.pl [--include-zero] symtab.txt num_extra > symtab_out.txt ";
}

if(@ARGV  == 3) {
    $include_zero = 1;
    $ARGV[0] eq "--include-zero" || die "Bad option/first argument $ARGV[0]";
    shift @ARGV;
} else {
    $include_zero = 0;
}

$input = $ARGV[0];
$nsyms = $ARGV[1];

open(F, "<$input") || die "Opening file $input";

while(<F>) {
    @A = split(" ", $_);
    @A == 2 || die "Bad line $_";
    $lastsym = $A[1];
    print;
}

if(!defined($lastsym)){
 die "Empty symbol file?";
}

if($include_zero) {
    $lastsym++;
    print "#0  $lastsym\n";
}

for($n = 1; $n <= $nsyms; $n++) {
    $y = $n + $lastsym;
    print "#$n  $y\n";
}
