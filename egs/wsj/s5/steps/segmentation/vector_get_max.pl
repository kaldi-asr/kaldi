#! /usr/bin/perl

use warnings;
use strict;

while (<>) {
    chomp;
    if (m/^\S+\s+\[.+\]\s*$/) {
        my @F = split;
        my $utt = shift @F;
        shift;
    
        my $max_id = 0;
        my $max = $F[0];
        for (my $i = 1; $i < $#F; $i++) {
            if ($F[$i] > $max) {
                $max_id = $i;
                $max = $F[$i];
            }
        }

        print "$utt $max_id\n";
    } else {
        die "Invalid line $_\n";
    }
}
