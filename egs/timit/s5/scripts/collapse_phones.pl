#!/usr/bin/perl
use strict ; 

my $ignore_first_field = 0;
if($ARGV[0] eq "--ignore-first-field") { $ignore_first_field = 1; shift @ARGV; }

my $symtab = shift @ARGV;

if(!defined $symtab) {
    die "Usage: collapse_phones.pl --ignore-first-field symtab [phoneme mapping] > output transcriptions\n";
}

my $mapping_str = shift @ARGV;
if(!defined $mapping_str) {
    die "Usage: collapse_phones.pl --ignore-first-field symtab [phoneme mapping] > output transcriptions\n";
}

my %mapping;
my @parts = split(",", $mapping_str);
for my $part (@parts) {
    my ($from, $to) = split(":", $part);
    $mapping{uc($from)} = uc($to) ; 
}

my %sym2int ; 
open(F, "<$symtab") || die "Error opening symbol table file $symtab";
while(<F>) {
    my @A = split(" ", $_);
    @A == 2 || die "bad line in symbol table file: $_";
    $sym2int{$A[0]} = $A[1] + 0;
}

# change the mappings. 
my %int2int ;
foreach my $key (keys %sym2int) {
   my $value = $sym2int{$key} ; 
   if (exists($mapping{$key})) {
      $int2int{$value} = $sym2int{$mapping{$key}} ;
   } else {
      $int2int{$value} = $value ; 
   }
}

while(<>) {
    my @A = split(" ", $_);
    if(@A == 0) {
        die "Empty line in transcriptions input.";
    }
    if($ignore_first_field) {
        my $key = shift @A;
        print $key . " ";
    }
    foreach $a (@A) {
        my $i = $int2int{$a};
        if(!defined ($i)) {
                die "collapse_phones.pl: undefined symbol $a\n";
        }
        print $i . " ";
    }
    print "\n";
}


