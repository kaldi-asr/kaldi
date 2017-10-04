#!/usr/bin/perl -w
# subs_check_oov.pl - find out of vocabulary words

use strict;
use warnings;
use Carp;

my $symtab = "data/lang/words.txt";
my $input = "data/local/tmp/subs/lm/es.txt";
my $o = "data/local/tmp/subs/lm/oovs.txt";

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
open my $O, '+>', $o or croak "problems with $o $!";

while ( my $line = <$I>) {
    chomp $line;
    my @A = split /\s/, $line;
    foreach my $a (@A) {
	if (!defined ($sym2int{$a})) {
            print $O "$a\n";
	}
    }
}
close $O;
close $I;
