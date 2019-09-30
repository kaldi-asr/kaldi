#!/usr/bin/env perl
# Copyright 2010-2012 Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

undef $field_begin;
undef $field_end;


if ($ARGV[0] eq "-f") {
  shift @ARGV;
  $field_spec = shift @ARGV;
  if ($field_spec =~ m/^\d+$/) {
    $field_begin = $field_spec - 1; $field_end = $field_spec - 1;
  }
  if ($field_spec =~ m/^(\d*)[-:](\d*)/) { # accept e.g. 1:10 as a courtesy (properly, 1-10)
    if ($1 ne "") {
      $field_begin = $1 - 1; # Change to zero-based indexing.
    }
    if ($2 ne "") {
      $field_end = $2 - 1; # Change to zero-based indexing.
    }
  }
  if (!defined $field_begin && !defined $field_end) {
    die "Bad argument to -f option: $field_spec";
  }
}
$symtab = shift @ARGV;
if(!defined $symtab) {
    print STDERR "Usage: int2sym.pl [options] symtab [input] > output\n" .
      "options: [-f (<field>|<field_start>-<field-end>)]\n" .
      "e.g.: -f 2, or -f 3-4\n";
    exit(1);
}

open(F, "<$symtab") || die "Error opening symbol table file $symtab";
while(<F>) {
    @A = split(" ", $_);
    @A == 2 || die "bad line in symbol table file: $_";
    $int2sym{$A[1]} = $A[0];
}

sub int2sym {
    my $a = shift @_;
    my $pos = shift @_;
    if($a !~  m:^\d+$:) { # not all digits..
      $pos1 = $pos+1; # make it one-based.
      die "int2sym.pl: found noninteger token $a [in position $pos1]\n";
    }
    $s = $int2sym{$a};
    if(!defined ($s)) {
      die "int2sym.pl: integer $a not in symbol table $symtab.";
    }
    return $s;
}

$error = 0;
while (<>) {
  @A = split(" ", $_);
  for ($pos = 0; $pos <= $#A; $pos++) {
    $a = $A[$pos];
    if ( (!defined $field_begin || $pos >= $field_begin)
         && (!defined $field_end || $pos <= $field_end)) {
      $a = int2sym($a, $pos);
    }
    print $a . " ";
  }
  print "\n";
}



