#!/usr/bin/perl -w
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
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


if (@ARGV > 0 && $ARGV[0] eq "-f") {
  shift @ARGV; 
  $field_spec = shift @ARGV; 
  if ($field_spec =~ m/^\d+$/) {
    $field_begin = $field_spec - 1; $field_end = $field_spec - 1;
  }
  if ($field_spec =~ m/^(\d*)[-:](\d*)/) { # accept e.g. 1:10 as a courtesty (properly, 1-10)
    if ($1 ne "") {
      $field_begin = $1 - 1;    # Change to zero-based indexing.
    }
    if ($2 ne "") {
      $field_end = $2 - 1;      # Change to zero-based indexing.
    }
  }
  if (!defined $field_begin && !defined $field_end) {
    die "Bad argument to -f option: $field_spec"; 
  }
}

# Mapping is obligatory
$permissive = 0;
if (@ARGV > 0 && $ARGV[0] eq '--permissive') {
  shift @ARGV;
  # Mapping is optional (missing key is printed to output)
  $permissive = 1;
}

if(@ARGV != 1) {
  print STDERR "Usage: apply_map.pl [options] map <input >output\n" .
    "options: [-f <field-range> ]\n" .
    "note: <field-range> can look like 4-5, or 4-, or 5-, or 1.\n" .
    "e.g.: echo A B | apply_map.pl a.txt\n" .
    "where a.txt is:\n" .
    "A a1 a2\n" .
    "B b\n" .
    "will produce:\n" .
    "a1 a2 b\n";
  exit(1);
}

($map) = @ARGV;
open(M, "<$map") || die "Error opening map file $map: $!";

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
    if ( (!defined $field_begin || $x >= $field_begin)
         && (!defined $field_end || $x <= $field_end)) {
      $a = $A[$x];
      if (!defined $map{$a}) {
        if (!$permissive) {
          die "apply_map.pl: undefined key $a\n"; 
        } else {
          print STDERR "apply_map.pl: warning! missing key $a\n";
        }
      } else {
        $A[$x] = $map{$a}; 
      }
    }
  }
  print join(" ", @A) . "\n";
}
