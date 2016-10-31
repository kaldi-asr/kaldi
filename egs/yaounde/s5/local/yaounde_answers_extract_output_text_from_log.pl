#!/usr/bin/perl -w
# yaounde_answers_extract_output_text_from_log.pl - get text from log file
use strict;
use warnings;
use Carp;
BEGIN {
    @ARGV == 1 or croak "yaounde_answers_extract_output_text_from_log.pl LOGFILE";
}

my $trigger = "ctell";

LINE: while ( my $line = <> ) {
    chomp $line;
    if ( $line =~ /^$trigger/ ) {
	my ($utterance, $text) = split /\s+/, $line, 2;
	print "$utterance\t$text\n";
    }
}
     
	 
     
