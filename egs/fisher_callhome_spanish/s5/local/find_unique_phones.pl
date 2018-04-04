#!/usr/bin/env perl
#Finds unique phones from the basic rules file
# Copyright 2014  Gaurav Kumar.   Apache 2.0

use utf8;

($b)=$ARGV[0];
($tmpdir)=$ARGV[1];
open(BB, "<", "$b/basic_rules") || die "Can't open basic rules";
binmode(BB, ":iso88591");
open(O, ">$tmpdir/phones") || die "Can't open text file for writing";
binmode(O, ":utf8");
my %phones = qw();
while (<BB>) {
  chomp;
  my @stringComponents = split(/\t/);
  m/->\s(\S+)/;
  my $phone = $1;
  $phone =~ tr/áéíóú/aeiou/;
  $phones{$phone} = 1;
}
foreach my $p (keys %phones) {
  print O $p, "\n";
}
#print keys %phones;
