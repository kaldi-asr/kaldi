#!/usr/bin/perl -w
#rename_wav_filenames.pl - copy wav file name to a unique name
use strict;
use warnings;
use Carp;

BEGIN {
    @ARGV == 2 or croak "USAGE: rename_wav_filenames.pl WAVFILENAMESFILE DIRECTORY";
}

use File::Copy;
    use File::Basename;

my ($wavfilenames,$dir) = @ARGV;

open my $WFN, '<', "$wavfilenames" or croak "could not open file $wavfilenames for reading $!";

mkdir $dir;

while ( my $line = <$WFN> ) {
    chomp $line;
    my $b = basename $line, ".wav";
    my ($speaker_num,$n) = split /\_/, $b, 2;
    $speaker_num =~ s/FR(\d\d\d)/$1/;
    my $d = dirname $line;
    my $speakername = basename $d;
    # create the new directory
    mkdir "$dir/$speakername";
    if ( $n < 10 ) {
$n = "000" . "$n";
} elsif  ( $n < 100 ) {
$n = "00" . "$n";
} elsif ( $n < 1000 ) {
$n = "0" . $n;
}
    copy "$line", "$dir/$speakername/FR${speakername}_$n.wav";
}


