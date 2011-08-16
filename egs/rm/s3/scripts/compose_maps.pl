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

# This program takes two arguments, which may be files or "-" for the
# standard input.  Both files must have lines with one or more fields,
# interpreted as a map from the first field (a string) to a list of strings.
# if the first file has as one of its lines
# A x y
# and the second has the lines
# x P
# y Q R
# then the output of this program will be
# A P Q R
# 
# Note that if x or y did not appear as the first field of file b, we would
# print a warning and omit the whole line rather than map it to the empty
# string.

if(@ARGV < 1 || @ARGV > 2 ) {
    die "Usage: compose_maps.pl map1 [map2] ";
}

$map1 = shift @ARGV;
open(I, "<$map1") || die "Opening first map $map1";

while(<>) { # <> represents map2.
    @A = split(" ", $_);
    if(@A == 0) { die "compose_maps.pl: invalid line in second map: $_\n"; }
    $key = shift @A;
    if(defined $map2{$key} ) { 
        print STDERR "compose_map.pl: key $key appears twice in second map.\n";
        if ($map2{$key} ne join(" ", @A)) {
            print STDERR " [and it has inconsistent values]\n";
        }
    }
    $map2{$key} = join(" ", @A);
}

while(<I>) {
    @A = split(" ", $_);
    if(@A == 0) { die "compose_map.pl: invalid line in second map: $_\n"; }
    $key = shift @A;
    $str = "$key ";
    $ok = 1;
    foreach $a (@A) {
        if(!defined $map2{$a}) { 
            print STDERR "compose_map.pl: key $a not defined in second map [skipping the line for $key]\n";
            $ok = 0;
        } else {
            $str = $str . "$map2{$a} ";
        }
    }
    if($ok) {
        print "$str\n";
    }
}
