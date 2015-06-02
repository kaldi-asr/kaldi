#!/usr/bin/env perl
# Copyright 2013 MERL (author: Felix Weninger)

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


use strict;

@ARGV == 1 || die "$0 <dot_file>\n";
my $dotfile = shift @ARGV;

open(F, "<$dotfile") || die "Error opening dot file $dotfile\n";
while(<F>) {
    $_ =~ m:(.+)\((\w{8})\)\s*$: || die "Bad line $_ in dot file $dotfile (line $.)\n";
    my $trans = $1;
    my $utt = $2;
    print "$utt $trans\n";
}
