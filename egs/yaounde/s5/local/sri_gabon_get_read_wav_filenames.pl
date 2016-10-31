#!/usr/bin/perl -w
# sri_gabon_get_wav_filenames.pl - get only wav files for read speech
use strict;
use warnings;
use Carp;

BEGIN {
    @ARGV == 1 or croak "USAGE: sri_gabon_get_read_wav_filenames.pl FILE
The input file contains a list of all the wav files
";
}

use File::Basename;

my @suffixlist = ( '.wav' );

LINE: while ( my $line = <> ) {
    chomp $line;
    my ($name,$path,$suffix) = fileparse($line,@suffixlist);
    my ( $afc_gabon, $Year_Month_Day, $SpeakerID, $read_conv, $fileNumber) = split /\_/, $name, 5;
    if ( $read_conv eq "read" ) {
	print "$line\n";
    }
}
