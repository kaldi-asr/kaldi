#!/usr/bin/perl -w
# get-lexicon4-lm.pl - get lm training data
use strict;
use warnings;
use Carp;

BEGIN {
    @ARGV == 1 or croak "USAGE: get-lexicon4lm.pl FILE";
}

while ( my $line = <> ) {
    chomp $line;
    my ($pth,$sent) = split /\t/, $line, 2;
print "$sent\n";
}

