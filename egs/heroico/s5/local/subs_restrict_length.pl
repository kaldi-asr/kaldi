#!/usr/bin/perl -w

# Copyright 2017 John Morgan
# Apache 2.0.

# subs_restrict_length.pl - restrict length of segments

use strict;
use warnings;
use Carp;

use Encode;

# set lower and upper bounds
my $lb = 8;
# only segments with at least  $lb words will be written
my $ub = 16;
# only segments with fewer than $ub words will be written

# input and output files
my $c = "data/local/tmp/subs/OpenSubtitles2016.en-es.es";
my $symtab = "data/lang/words.txt";
my $rl = "data/local/tmp/subs/lm/es.txt";
my $oo = "data/local/tmp/subs/lm/oovs.txt";
my $iv = "data/local/tmp/subs/lm/in_vocabulary.txt";

open my $C, '<', $c or croak "problems with $c $!";

system "mkdir -p data/local/tmp/subs/lm";

open my $RL, '+>:utf8', $rl or croak "problems with $rl $!";

LINE: while ( my $line = <$C> ) {

    $line = decode_utf8 $line;

    chomp $line;

    my @tokens = split /\s+/, $line;

    next LINE if ( ($#tokens < $lb) or ($#tokens > $ub ));

    my @lc_tokens = ();
    for my $t (@tokens) {
	push @lc_tokens, lc $t;
    }

    print $RL "@lc_tokens\n";

}

close $C;
close $RL;

# find out of vocabulary words

# $symtab points to a file containing a map of symbols to integers

# hash for word to integer map
my %sym2int = ();

open my $F, '<', $symtab or croak "problem with $symtab $!";

# store words to int map in hash
while( my $line = <$F>) {
    chomp $line;
    my ($s,$i) = split /\s/, $line, 2;
    $sym2int{$s} = $i;
}
close $F;

open my $I, '<', $rl or croak "problem with $rl $!";
open my $OO, '+>', $oo or croak "problems with $oo $!";

while ( my $line = <$I>) {
    chomp $line;
    my @A = split /\s/, $line;
    foreach my $a (@A) {
	if (!defined ($sym2int{$a})) {
            print $OO "$a\n";
	}
    }
}
close $OO;
close $I;

# remove segments with OOVs

# store OOVS in hash
my %oov = ();
open my $V, '<', $oo or croak "problems with $oo $!";
while ( my $line = <$V> ) {
    chomp $line;
    $oov{$line} = 1;
}
close $V;

open my $L, '<', $rl or croak "problems with $rl $!";
open my $IV, '+>', $iv or croak "problems with $iv $!";

SEGMENT: while ( my $segment = <$L> ) {
    chomp $segment;
    my @words = split /\s+/, $segment;
    foreach my $word ( sort @words ) {
	next SEGMENT if ( $oov{$word} );
    }
    print $IV "$segment\n";
}
close $IV;
close $L;
