#!/usr/bin/env perl
#Finds unique phones from the basic rules file

use utf8;

($b)=$ARGV[0];
($tmpdir)=$ARGV[1];
open(BB, "<", "$b/basic_rules") || die "Can't open basic rules";
open(O, ">$tmpdir/phones") || die "Can't open text file for writing";
binmode(O, ":utf8");
my %phones = qw();
while (<BB>) {
  chomp;
  my @stringComponents = split(/\t/);
  m/->\s(\S+)/;
  $phones{$1} = 1;
}
foreach my $p (keys %phones) {
  print O $p, "\n";
}
#print keys %phones;
