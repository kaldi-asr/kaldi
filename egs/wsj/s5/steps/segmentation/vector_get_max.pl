#! /usr/bin/perl

# Copyright 2016  Vimal Manohar
# Apache 2.0

# This script reads an archive of vectors in text format and 
# writes an archive of the maximum element of the vector indexed by the 
# same key.

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
