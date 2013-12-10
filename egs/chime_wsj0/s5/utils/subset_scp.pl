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

# This program selects a subset of N elements in the scp.

# By default, it selects them evenly from throughout the scp, in order to avoid
# selecting too many from the same speaker.  It prints them on the standard
# output.
# With the option --first, it just selects the N first utterances.
# With the option --last, it just selects the N last utterances.

# Last modified by JHU & HKUST @2013


$first = 0;
$last = 0;
if ($ARGV[0] eq "--first") {
  shift;
  $first = 1;
}
if ($ARGV[0] eq "--last") {
  shift;
  $last = 1;
}

if(@ARGV < 2 ) {
    die "Usage: subset_scp.pl [--first|--last] N in.scp ";
}

$N = shift @ARGV;
if($N == 0) {
    die "First command-line parameter to subset_scp.pl must be an integer, got \"$N\"";
}
$inscp = shift @ARGV;
open(I, "<$inscp") || die "Opening input scp file $inscp";

@F = ();
while(<I>) {
    push @F, $_;
}
$numlines = @F;
if($N > $numlines) {
    die "You requested from subset_scp.pl more elements than available: $N > $numlines";
}

sub select_n {
    my ($start,$end,$num_needed) = @_;
    my $diff = $end - $start;
    if($num_needed > $diff) { die "select_n: code error"; }
    if($diff == 1 ) {
        if($num_needed  > 0) {
            print $F[$start];
        }
    } else {
        my $halfdiff = int($diff/2);
        my $halfneeded = int($num_needed/2);
        select_n($start, $start+$halfdiff, $halfneeded);
        select_n($start+$halfdiff, $end, $num_needed - $halfneeded);
    }
}

if ( ! $first && ! $last) {
  select_n(0, $numlines, $N);
} else {
  if ($first) { # --first option: same as head.
    for ($n = 0; $n < $N; $n++) {
      print $F[$n];
    }
  } else { # --last option: same as tail.
    for ($n = @F - $N; $n < @F; $n++) {
      print $F[$n];
    }
  }
}
