#!/usr/bin/perl -w
#get_utt2wavfilename.pl - write wav.scp file
use strict;
use warnings;
use Carp;

BEGIN {
    @ARGV == 2 or croak "USAGE: get_utt2wavfilename.pl WAVPATH FOLDSPEAKERNAMES";
}

use File::Basename;

my ($wavpath,$spk) = @ARGV;

open my $SPK, '<', "$spk" or croak "could not open file $spk for reading $!";

while  ( my $s = <$SPK> ) {
    chomp $s;
    
    opendir my $SPKD, "$wavpath/$s" or croak "could not open directory $wavpath/$s for reading $!";
    while ( my $entry = readdir $SPKD ) {
	if ( $entry =~ /.wav$/ ) {
	    my $utt = basename $entry, ".wav";
	    print "$utt\t$wavpath/$s/$entry\n";
	}
    }
    
}

