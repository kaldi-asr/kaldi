#!/usr/bin/perl -w
#gp_rename_wav_filenames.pl - copy wav file name to a unique name
use strict;
use warnings;
use Carp;

BEGIN {
    @ARGV == 2 or croak "USAGE: gp_rename_wav_filenames.pl WAVFILENAMESFILE DIRECTORY
Assumes the first argument contains a list of file names ending in .wav
like:
/path/FR001_0001.wav
";
}

use File::Copy;
    use File::Basename;

my ($wavfilenames,$dir) = @ARGV;

my @suffixlist = ( '.wav' );
open my $WFN, '<', "$wavfilenames" or croak "could not open file $wavfilenames for reading $!";

mkdir $dir;

while ( my $line = <$WFN> ) {
    chomp $line;
    my ($name,$path,$suffix) = fileparse($line,@suffixlist);
    my ( $afc_gabon, $Year_Month_Day, $SpeakerID, $read_conv, $file_number) = split /\_/, $name, 5;
    my $speakername = basename $path;
    # create the new directory
    mkdir "$dir/sri_gabon_$speakername";
    copy "$line", "$dir/sri_gabon_$speakername/sri_gabon_${speakername}_${read_conv}_${file_number}.wav";    
}


