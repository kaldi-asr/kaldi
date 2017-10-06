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
my $ub = 16;

# input and output files
my $c = "data/local/tmp/subs/OpenSubtitles2016.en-es.es";
my $symtab = "data/lang/words.txt";
my $input = "data/local/tmp/subs/lm/es.txt";
my $o = "data/local/tmp/subs/lm/es.txt";
my $oo = "data/local/tmp/subs/lm/oovs.txt";
my $l = "data/local/tmp/subs/lm/es.txt";
my $v = "data/local/tmp/subs/lm/oovs.txt";
my $iv = "data/local/lm/subs_es_in_vocabulary.txt";

open my $C, '<', $c or croak "problems with $c $!";

system "mkdir -p data/local/tmp/subs/lm";

open my $O, '+>:utf8', $o or croak "problems with $o $!";

LINE: while ( my $line = <$C> ) {

    $line = decode_utf8 $line;

    chomp $line;

    my @tokens = split /\s+/, $line;

    next LINE if ( ($#tokens < $lb) or ($#tokens > $ub ));

    print $O "$line\n";

}

close $C;
close $O;

# find out of vocabulary words

# $symtab points to a file containing a map of symbols to integers
# $input points to a file containing  the text processed in this script

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

open my $I, '<', $input or croak "problem with $input $!";
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
open my $V, '<', $v or croak "problems with $v $!";
while ( my $line = <$V> ) {
    chomp $line;
    $oov{$line} = 1;
}
close $V;

open my $L, '<', $l or croak "problems with $l $!";
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
