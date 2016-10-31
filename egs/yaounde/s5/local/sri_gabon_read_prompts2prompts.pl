#!/usr/bin/perl -w
#sri_gabon_read_prompts2prompts.pl - make promt files for each speaker
use strict;
use warnings;
use Carp;

BEGIN {
    @ARGV == 2 or croak "USAGE: sri_gabon_read_prompts2prompts.pl WAVFILELIST PROMPTSLIST
output:
1 file per speaker directory
The file is named prompts
1 line per prompt
2 fields on each line
First field is the path to the wav file 
the extention .wav is not written  
The second field is the prompt
";
}

use File::Basename;

my ($wfile,$promptsfile) = @ARGV;

mkdir "data/prompts";
open my $SPF, '<', $promptsfile or croak "could not open file $promptsfile for reading $!";

# store the prompts in a hash 
my %prompts = ();
while ( my $line = <$SPF>) {
    chomp $line;
    my ($num,$sent) = split /\s/, $line, 2;
$prompts{$num} = $sent;    
}
close $SPF;

open my $WFNFF, '<', $wfile or croak "could not open file $wfile for reading $!";
# write the speaker prompt files
while ( my $line = <$WFNFF> ) {
    chomp $line;
    my $b = basename $line, ".wav";
    my $d = dirname $line;
    my $spkdir = basename $d;
    mkdir "data/prompts/$spkdir";

    open my $PF, '+>>', "data/prompts/$spkdir/prompts" or croak "could not open file prompts/$spkdir/data/prompts for appending $!";
    print $PF "data/prompts/$spkdir/$b\t$prompts{$b}\n";
    close $PF;
}
close $WFNFF;
