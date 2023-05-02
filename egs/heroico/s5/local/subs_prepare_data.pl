#!/usr/bin/env perl

# Copyright 2017 John Morgan
# Apache 2.0.

# subs_prepare_data.pl - condition subs data for lm training

use strict;
use warnings;
use Carp;

use Encode;

# set lower and upper bounds
my $low_bound = 8;
# only segments with at least  $low_bound words will be written
my $up_bound = 16;
# only segments with fewer than $up_bound words will be written

# input and output files

my $corpus = "OpenSubtitles.en-es.es";
my $symbol_table = "data/lang/words.txt";
my $filtered = "data/local/tmp/subs/lm/es.txt";
my $oovs = "data/local/tmp/subs/lm/oovs.txt";
my $iv = "data/local/tmp/subs/lm/in_vocabulary.txt";

open my $C, '<', $corpus or croak "problems with $corpus $!";

system "mkdir -p data/local/tmp/subs/lm";

if ( -e $filtered ) {
    warn "$filtered already exists.";
} else {
  open my $FLT, '+>:utf8', $filtered or croak "problems with $filtered $!";
  LINE: while ( my $line = <$C> ) {
      $line = decode_utf8 $line;
      chomp $line;

      my @tokens = split /\s+/, $line;

      next LINE if ( ($#tokens < $low_bound) or ($#tokens > $up_bound ));

      # remove punctuation
      $line =~ s/(\p{Punctuation}+|\p{Dash_Punctuation}+|\p{Close_Punctuation}+|\p{Open_Punctuation}+|\p{Initial_Punctuation}+|\p{Final_Punctuation}+|\p{Connector_Punctuation}+|\p{Other_Punctuation}+|[	 ]+)/ /msxg;
      #convert tabs to white space
      $line =~ s/\t/ /g;
      #hard to soft space
      $line =~ s/ / /g;
      #squeeze white space
      $line =~ s/\s+/ /g;
      #initial and final white space
      $line =~ s/^\p{Separator}+//;
      $line =~ s/\p{Separator}+$//;
      #down case
      $line = lc $line;

      print $FLT "$line\n";
  }
  close $FLT;
}
close $C;


# find out of vocabulary words

# $symbol_table points to a file containing a map of symbols to integers

# hash for word to integer map
my %sym2int = ();

open my $F, '<', $symbol_table or croak "problem with $symbol_table $!";

# store words to int map in hash
while( my $line = <$F>) {
    chomp $line;
    my ($s,$i) = split /\s/, $line, 2;
    $sym2int{$s} = $i;
}
close $F;

open my $I, '<', $filtered or croak "problem with $filtered $!";
open my $OOVS, '+>', $oovs or croak "problems with $oovs $!";

while ( my $line = <$I>) {
    chomp $line;
    my @A = split /\s/, $line;
    foreach my $a (@A) {
	if (!defined ($sym2int{$a})) {
            print $OOVS "$a\n";
	}
    }
}
close $OOVS;
close $I;

# remove segments with OOVs

# store OOVS in hash
my %oov = ();
open my $V, '<', $oovs or croak "problems with $oovs $!";
while ( my $line = <$V> ) {
    chomp $line;
    $oov{$line} = 1;
}
close $V;

open my $L, '<', $filtered or croak "problems with $filtered $!";
open my $IV, '+>', $iv or croak "problems with $iv $!";

SEGMENT: while ( my $segment = <$L> ) {
    chomp $segment;
    my @words = split /\s+/, $segment;
    foreach my $word ( sort @words ) {
	next SEGMENT if ( $oov{$word} );
    }
    print $IV "$segment\n";
}
close $IV;
close $L;
