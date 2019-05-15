#!/usr/bin/env perl

# Copyright 2014  Guoguo Chen
#           2014  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0.
#
use strict;
use warnings;
use Getopt::Long;

my $Usage = <<EOU;
This script reads the Arpa format language model, and maps the words into
integers or vice versa. It ignores the words that are not in the symbol table,
and updates the head information.

It will be used joinly with lmbin/arpa-to-const-arpa to build ConstArpaLm format
language model. We first map the words in an Arpa format language model to
integers, and then use lmbin/arpa-to-const-arpa to build a ConstArpaLm format
language model.

Usage: utils/map_arpa_lm.pl [options] <vocab-file> < input-arpa >output-arpa
 e.g.: utils/map_arpa_lm.pl words.txt <arpa_lm.txt >arpa_lm.int

Allowed options:
  --sym2int   : If true, maps words to integers, other wise maps integers to
                words. (boolean, default = true)

EOU

my $sym2int = "true";
GetOptions('sym2int=s' => \$sym2int);

($sym2int eq "true" || $sym2int eq "false") ||
  die "$0: Bad value for option --sym2int\n";

if (@ARGV != 1) {
  die $Usage;
}

# Gets parameters.
my $symtab = shift @ARGV;
my $arpa_in = shift @ARGV;
my $arpa_out = shift @ARGV;

# Opens files.
open(M, "<$symtab") || die "$0: Fail to open $symtab\n";

# Reads in the mapper.
my %mapper;
while (<M>) {
  chomp;
  my @col = split(/[\s]+/, $_);
  @col == 2 || die "$0: Bad line in mapper file \"$_\"\n";
  if ($sym2int eq "true") {
    if (defined($mapper{$col[0]})) {
      die "$0: Duplicate entry \"$col[0]\"\n";
    }
    $mapper{$col[0]} = $col[1];
  } else {
    if (defined($mapper{$col[1]})) {
      die "$0: Duplicate entry \"$col[1]\"\n";
    }
    $mapper{$col[1]} = $col[0];
  }
}

my $num_oov_lines = 0;
my $max_oov_warn = 20;

# Parses Arpa n-gram language model.
my $arpa = "";
my $current_order = -1;
my %head_ngram_count;
my %actual_ngram_count;
while (<STDIN>) {
  chomp;
  my @col = split(" ", $_);

  if ($current_order == -1 and ! m/^\\data\\$/) {
    next;
  }

  if (m/^\\data\\$/) {
    print STDERR "$0: Processing \"\\data\\\"\n";
    print "$_\n";
    $current_order = 0;
  } elsif (m/^\\[0-9]*-grams:$/) {
    $current_order = $_;
    $current_order =~ s/-grams:$//g;
    $current_order =~ s/^\\//g;
    print "$_\n";
    print STDERR "$0: Processing \"\\$current_order-grams:\\\"\n";
  } elsif (m/^\\end\\/) {
    print "$_\n";
  } elsif ($_ eq "") {
    if ($current_order >= 1) {
      print "\n";
    }
  } else {
    if ($current_order == 0) {
      # echo head section.
      print "$_\n";
    } else {
      # Parses n-gram section.
      if (@col > 2 + $current_order || @col < 1 + $current_order) {
        die "$0: Bad line in arpa lm \"$_\"\n";
      }
      my $prob = shift @col;
      my $is_oov = 0;
      for (my $i = 0; $i < $current_order; $i++) {
        my $temp = $mapper{$col[$i]};
        if (!defined($temp)) {
          $is_oov = 1;
          $num_oov_lines++;
          last;
        } else {
          $col[$i] = $temp;
        }
      }
      if (!$is_oov) {
        my $rest_of_line = join(" ", @col);
        print "$prob\t$rest_of_line\n";
      } else {
        if ($num_oov_lines < $max_oov_warn) {
          print STDERR "$0: Warning: OOV line $_\n";
        }
      }
    }
  }
}

if ($num_oov_lines > 0) {
  print STDERR "$0: $num_oov_lines lines of the Arpa file contained OOVs and ";
  print STDERR "were not printed.\n";
}

close(M);
