#!/usr/bin/perl -w
#get_gp_prompts.pl - make 2 parts of a prompts file
use strict;
use warnings;
use Carp;
BEGIN {
    @ARGV == 1 or croak "USAGE: get_gp_prompts.pl TRLLISTFILE
output:
2 files
1 file with 1 line per prompt
1   file with 1 prompt id per line
The prompt id consists of a speaker id and the prompt number
";
}

my $out_id_file = "data/local/tmp/id_list.txt";
my $out_prompts_file = "data/local/tmp/prompts_list.txt";

my ($flist) = @ARGV;

open my $L, '<', "$flist" or croak "could not open file $flist for reading $!";
open my $I, '+>', "$out_id_file" or croak "could not open file $out_id_file for writing $!";
open my $P, '+>', "$out_prompts_file" or croak "could not open file $out_prompts_file for writing $!";

while ( my $line = <$L> ) {
    chomp $line;
    open my $T, '<', "$line" or croak "could not open file $line for reading $!";
    my $spkr = "";
    while ( my $linea = <$T> ) {
	chomp $linea;
	if ( $linea =~ /^\;SprecherID\s(FR\d{1,3})/ ) {
	    $spkr = $1;
	} elsif ( $linea =~ /^\;\s(\d{1,})/ ) {
	    my $n = $1;
	    if ( $n < 10 ) {
		print $I "$spkr\_000$n\n";
	    } elsif ( $n < 100 ) {
		print $I "$spkr\_00$n\n";
	    } elsif ( $n < 1000 ) {
		print $I "$spkr\_0$n\n";
	    } elsif ( $n < 10000 )  {
		print $I "$spkr\_$n\n";
	    }
	} else {
	    print $P "$linea\n";
	}      
    }
    close $T;
}

close $I;
close $P;

