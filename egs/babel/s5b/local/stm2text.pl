#!/usr/bin/env perl

# Copyright 2012  Johns Hopkins University (Author: Yenda Trmal)
# Apache 2.0.

#This script takes the source STM file and generates the *.txt files which 
#are usually part of the BABEL delivery
#The *.txt files are not the part of the delivery for the evalpart1 subset
#The program works as a filter and the only parameter it expects is
#the path to the output directory
#The filenames are figured out from the STM file
#example of usage:
#  cat data/evalpart1/stm local/stm2text.pl data/raw_evalpart1_data/transcriptions

use strict; 
use warnings;

use utf8;
use Data::Dumper;

binmode(STDIN, ":encoding(utf8)");
binmode(STDOUT, ":encoding(utf8)");

my $output_dir = $ARGV[0];
my $prev_filename = "";
my $OUTPUT;
while ( <STDIN> ) {
  chop;
  my ($filename, $channel, $speaker, $start, $end, $text) = split(" ", $_, 6);
  next if ( $filename =~ /;;.*/ );
  #$filename =~ s/;;(.*)/$1/ if ( $filename =~ /;;.*/ );
  $text = "<no-speech>" if not $text;
   
  if ( $prev_filename ne $filename ) {
    #close($OUTPUT) if ( tell(FH) != -1 );
    print "$output_dir/$filename.txt\n";
    open($OUTPUT, ">:encoding(UTF-8)", "$output_dir/$filename.txt") or die $!;
    $prev_filename = $filename;
  }

  print $OUTPUT "[$start]\n";
  print $OUTPUT "$text\n";
}
