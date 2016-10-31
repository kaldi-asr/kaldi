#!/usr/bin/perl -w
#get-lexicon.pl - get a list of words from the training sentences
use strict;
use warnings;
use Carp;
while ( my $line = <> ) {
    chomp $line;
    my @tokens = split /\s+/, $line;
    for my $token (sort @tokens) {
	if ( $token ne "" ){
	    print "$token\n";;
	}
    }
}

