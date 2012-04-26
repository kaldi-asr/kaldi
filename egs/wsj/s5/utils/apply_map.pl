#!/usr/bin/perl -w
# Copyright 2012  Daniel Povey
# Apache 2.0.

# This program is a bit like ./sym2int.pl in that it applies a map
# to things in a file, but it's a bit more general in that it doesn't
# assume the things being mapped to are single tokens, they could
# be sequences of tokens.  

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

if(@ARGV != 1) {
  print STDERR "Usage: apply_map.pl map <input >output\n" .
    "e.g.: echo A B | apply_map.pl <a.txt\n" .
    "where a.txt is:\n" .
    "A a1 a2\n" .
    "B b\n" .
    "will produce:\n" .
    "a1 a2 b\n";
}

($map) = @ARGV;
open(M, "<$map") || die "Opening map file $map";

while (<M>) {
  @A = split(" ", $_);
  @A >= 1 || die "apply_map.pl: empty line.";
  $i = shift @A;
  $o = join(" ", @A);
  $map{$i} = $o;
}

while(<STDIN>) {
  @A = split(" ", $_);
  for ($x = 0; $x < @A; $x++) {
    $a = $A[$x];
    if (!defined $map{$a}) { die "compose_maps.pl: undefined key $a\n"; }
    $A[$x] = $map{$a};
  }
  print join(" ", @A) . "\n";
}
