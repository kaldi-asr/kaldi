#!/usr/bin/env perl
# Copyright 2012  Johns Hopkins University (Author: Yenda Trmal)
# Apache 2.0.

use Data::Dumper;
$Data::Dumper::Indent = 1;

binmode STDOUT, ":utf8"; 
binmode STDIN, ":utf8"; 

sub permute {

    my $last = pop @_;

    unless(@_) {
           return map([$_], @$last);
    }

    return map { 
                 my $left = $_; 
                 map([@$left, $_], @$last)
               } 
               permute(@_);
}

$oov_count=0;

$ignore_oov = 0;
$ignore_first_field = 0;
for($x = 0; $x < 2; $x++) {
  if ($ARGV[0] eq "--map-oov") {
    shift @ARGV; $map_oov = shift @ARGV;
  }
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
#print Dumper(\%sym2int);

if (defined $map_oov && $map_oov !~ m/^\d+$/) { # not numeric-> look it up
  if (!defined $sym2int{$map_oov}) { die "OOV symbol $map_oov not defined."; }
  $map_oov = $sym2int{$map_oov};
}

$lines=0;
while (<>) {
  @A = split(" ", $_);
  @B = ();
  $lines = $lines + 1;
  $undefined_words = 0;
  for ($n = 1; $n < @A; $n++) {
    $a = $A[$n];
    $i = $sym2int{$a};
    if (!defined ($i)) {
      if (defined $map_oov) {
        if ($num_warning++ < $max_warning) {
          print STDERR "sym2int.pl: replacing $a with $map_oov\n";
          if ($num_warning == $max_warning) {
            print STDERR "sym2int.pl: not warning for OOVs any more times\n";
          }
        }
        $i = [ $map_oov ];
      } else {
        $pos = $n+1;
        die "sym2int.pl: undefined symbol $a (in position $pos)\n";
      }
      $undefined_words = $undefined_words + 1;
    }
    $a = $i;
    push @B, $a;
  }
    #if ( defined $sym2int{$A[$n]} ) {
    #  push @B, $sym2int{$A[$n]};
    #} else {
    #  push @B, [0];
    #}
  if ($undefined_words > 0) {
    $oov_count = $oov_count + 1;
  }
  @C = permute @B;
  #print Dumper(\@B);
  #print Dumper(\@C);
  foreach $phrase ( @C ) {
    print "$A[0] ";
    print join(" ", @{$phrase});
    print "\n";
  }
}

print STDERR "Remaped/ignored $oov_count phrases...\n";

