#!/usr/bin/perl -w
# lexicon2lexiconp.pl - put 1.0 for the pronunciation probability in the lexicon
use strict;
use warnings;
use Carp;

BEGIN {
    @ARGV == 1 or croak "USAGE: lexicon2lexiconp.pl LEXICON";
}

while ( my $line = <> ) {
    chomp $line;
    my ($w,$p) = split /\s/, $line, 2;
    if ( not defined $p ) {
	croak "$line";
    }
    print "$w 1.0\t$p\n";
}
