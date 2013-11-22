#!/usr/bin/perl
# Copyright 2012  Johns Hopkins University (Author: Yenda Trmal)
# Apache 2.0.

use Data::Dumper;
$Data::Dumper::Indent = 1;

binmode STDOUT, ":utf8"; 
binmode STDIN, ":utf8"; 

$ignore_oov = 0;
$ignore_first_field = 0;
for($x = 0; $x < 2; $x++) {
  if ($ARGV[0] eq "-f") {
    shift @ARGV; 
    $field_spec = shift @ARGV; 
    if ($field_spec =~ m/^\d+$/) {
      $field_begin = $field_spec - 1; $field_end = $field_spec - 1;
    }
    if ($field_spec =~ m/^(\d*)[-:](\d*)/) { # accept e.g. 1:10 as a courtesty (properly, 1-10)
      if ($1 ne "") {
        $field_begin = $1 - 1;  # Change to zero-based indexing.
      }
      if ($2 ne "") {
        $field_end = $2 - 1;    # Change to zero-based indexing.
      }
    }
    if (!defined $field_begin && !defined $field_end) {
      die "Bad argument to -f option: $field_spec"; 
    }
  }
}

$symtab = shift @ARGV;
if (!defined $symtab) {
  print STDERR "Usage: sym2int.pl [options] symtab [input transcriptions] > output transcriptions\n" .
    "options: [--map-oov <oov-symbol> ]  [-f <field-range> ]\n" .
      "note: <field-range> can look like 4-5, or 4-, or 5-, or 1.\n";
}


open(F, "<:encoding(UTF-8)", $symtab) || die "Error opening symbol table file $symtab";
while(<F>) {
    @A = split(" ", $_);
    @A == 2 || die "bad line in symbol table file: $_";
    
    if ( not defined( $sym2int{$A[0]} ) ) {
      $sym2int{$A[0]} = [];
    }
    push @{ $sym2int{$A[0]} }, $A[1] + 0;
}


$lines=0;
while (<>) {
  @A = split(" ", $_);
  @B = ();
  for ($n = 0; $n < @A; $n++) {
    if ( (!defined $field_begin || $n >= $field_begin)
      && (!defined $field_end || $n <= $field_end)) {
      $a = $A[$n];
      $i = $sym2int{$a};
      if (!defined ($i)) {
        print $a . "\n";
      } 
    }
  }
}


