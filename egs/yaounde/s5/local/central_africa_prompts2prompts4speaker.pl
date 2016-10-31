#!/usr/bin/perl -w
#central_africa_prompts2prompts4speaker.pl - make promt files for each speaker
use strict;
use warnings;
use Carp;

BEGIN {
    @ARGV == 2 or croak "USAGE: central_africa_prompts2prompts4speaker.pl WAVFILELIST PROMPTSLIST
Input: 2 filenames
The first file contains all the .wav file names
The second file contains all the prompts.
The second file has 2 fields:
A prompt number
The prompt itself.
output:
1 file per speaker directory
The file is named prompts
1 line per prompt
2 fields on each line
First field is the path to the wav file 
the extention .wav is not written  
The second field is the prompt
All the prompt files are written under a directory called prompts.
";
}

use File::Spec;
use File::Basename;

my ($wfile,$promptsfile) = @ARGV;

system "mkdir -p data/prompts";
open my $SPF, '<', $promptsfile or croak "could not open file $promptsfile for reading $!";

# store the prompts in a hash 
my %prompts = ();
while ( my $line = <$SPF>) {
    chomp $line;
    my ($num,$sent) = split /\t/, $line, 2;
$sent =~ s/^\s+//;
$sent =~ s/\s+$//;
$prompts{$num} = $sent;    
}
close $SPF;

open my $WFNFF, '<', $wfile or croak "could not open file $wfile for reading $!";
# write the speaker prompt files
while ( my $path = <$WFNFF> ) {
    chomp $path;
    my ($volume,$directories,$file) = File::Spec->splitpath( $path );
    my $b = basename $path, ".wav";
    my ($country,$gender,$speaker_num,$bn) = split /\_/, $b, 4;
    my @dirs = File::Spec->splitdir( $directories );
    system "mkdir -p data/prompts/${country}_${gender}_${speaker_num}";

    open my $PF, '+>>', "data/prompts/${country}_${gender}_${speaker_num}/prompts" or croak "could not open file data/prompts/${country}_${gender}_${speaker_num}/prompts for appending $!";
    print $PF "data/prompts/${country}_${gender}_${speaker_num}/$b\t$prompts{$b}\n";
    close $PF;
}
close $WFNFF;

