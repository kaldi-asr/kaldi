#!/usr/bin/perl -w
# subs_remove_oov_segments.pl - remove segments with OOVs

use strict;
use warnings;
use Carp;

my $l = "data/local/tmp/subs/lm/es.txt";
my $v = "data/local/tmp/subs/lm/oovs_uniq.txt";
my $o = "data/local/lm/subs_es_in_vocabulary.txt";

# store OOVS in hash
my %oov = ();
open my $V, '<', $v or croak "problems with $v $!";
while ( my $line = <$V> ) {
    chomp $line;
    $oov{$line} = 1;
}
close $V;

open my $L, '<', $l or croak "problems with $l $!";
open my $O, '+>', $o or croak "problems with $o $!";
SEGMENT: while ( my $segment = <$L> ) {
    chomp $segment;
    my @words = split /\s+/, $segment;
    foreach my $word ( sort @words ) {
	next SEGMENT if ( $oov{$word} );
    }
    print $O "$segment\n";
}
close $O;
close $L;
