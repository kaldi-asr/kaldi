#!/usr/bin/env perl
#===============================================================================
# Copyright (c) 2017  Johns Hopkins University (Author: Jan "Yenda" Trmal <jtrmal@gmail.com>)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

use strict;
use warnings;
use utf8;

require HTML::Parser or die "This script needs HTML::Parser from CPAN";
HTML::Parser->import();

binmode(STDOUT, ":utf8");

sub  trim { my $s = shift; $s =~ s/^\s+|\s+$//g; return $s };

sub parse_sgml_tag {
  my $tag = shift(@_);
  my %ret;
  
  if ($tag !~ /=/) {
    return %ret;
  }
  
  $tag =~ s/<[a-zA-Z]+ //;
  $tag =~ s/> *$//;
  #print $tag . "\n";

  my @key_value_pairs = split / *,? +/, $tag;
  for my $entry(@key_value_pairs) {
    (my $key, my $value) = split '=', $entry, 2;
    $ret{$key}=$value;
  }
  return %ret;
}

if (@ARGV != 1) {
  print STDERR "$0: This script needs exactly one parameter (list of SGML files)\n";
  print STDERR "  Usage: $0 <transripts>\n";
  print STDERR "  where\n";
  print STDERR "    <transcripts> is a file containing the official SGML format\n";
  print STDERR "      transcripts. The files are parsed and the parsed representation\n";
  print STDERR "      is dumped to STDOUT (one utterance + the additional data fields\n";
  print STDERR "      per line (we dump all the fields, but not all fields are used\n";
  print STDERR "      in the recipe).\n";
  die;
}
my $filelist=$ARGV[0];

my $p = HTML::Parser->new();

my @files=();
open(F, '<', $filelist) or die "Could not open file $filelist: $?\n";
while(<F>) {
  chomp;
  push @files, $_;
}

foreach my $file (@files) {
  my $reporter="";
  my $start = -1;
  my $end = -1;
  my $turn_start = -1;
  my $turn_end = -1;
  my $turn_speaker;
  my $turn_dialect = "unknown";
  my $turn_sex;
  my $section_start = -1;
  my $section_end = -1;
  my $filename = "";
  my $seq = 0;
  my @text = ();
  my $time;
  my @tagqueue;

  my $sgml_file = `basename $file`;
  $sgml_file = trim $sgml_file;
  $sgml_file =~ s/\.sgm$//g;

  open(my $f, '<:encoding(iso-8859-1)', $file) or die "Could not open file $file: $?\n";

  while(my $line = <$f>) {
    chomp $line;
    $line = trim $line;
    next unless $line;

    if ($line =~ /<episode/) {
      my %tags = parse_sgml_tag $line;
      $filename = $tags{'filename'};
      $filename =~ s/"//g;

      print STDERR "$0: WARNING: SGML filename does not match episode filename $filename in file $file\n";
      #print "BS: $line\n";
      push @tagqueue, ["episode", \%tags];
      ;
    } elsif ($line =~ /<\/episode/) {
      my $p = pop @tagqueue;
      $line =~ s/<\/(.*)( +.*)?>/$1/g;
      die "Unaligned tags: " . $p->[0] . " vs $line" if $p->[0] ne $line;
      #print "ES: $line\n";
      ;
    } elsif ($line =~ /<section/) {
      my %tags = parse_sgml_tag $line;
      $section_start = $tags{'startTime'};
      $section_end = $tags{'endTime'};
      #print "BS: $line\n";
      push @tagqueue, ["section", \%tags];
      ;
    } elsif ($line =~ /<\/section/) {
      my $p = pop @tagqueue;
      $line =~ s/<\/(.*)( +.*)?>/$1/g;
      die "Unaligned tags: " . $p->[0] . " vs $line" if $p->[0] ne $line;
      #print "ES: $line\n";
      ;
    } elsif ($line =~ /<turn/) { 
      #print "BT: $line\n";
      my %tags = parse_sgml_tag $line;
      $turn_speaker = $tags{'speaker'};
      $turn_start = $tags{'startTime'};
      $turn_end = $tags{'endTime'};
      $turn_dialect = $tags{'dialect'} if $tags{'dialect'};
      $turn_sex = $tags{'sex'};
      push @tagqueue, ["turn", \%tags];
      ;
    } elsif ($line =~ /<\/turn/) {
      my $p = pop @tagqueue;
      $line =~ s/<\/(.*)( +.*)?>/$1/g;
      die "Unaligned tags: " . $p->[0] . " vs $line" if $p->[0] ne $line;

      #print join(" ", @text) . "\n" if @text > 0;
      my $new_time = $turn_end;
      if (@text > 0) {
        print "$sgml_file $filename $turn_speaker $turn_dialect $turn_sex $time $new_time ";
        print join(" ", @text) . "\n";
      }
      @text = ();
      $time = 0;
      $turn_speaker = "XXX";
      $turn_start = "XXX";
      $turn_end = "XXX";
      $turn_dialect = "XXX";
      $turn_sex = "XXX";
      #print "ET: $line\n";
      ;
    } elsif ($line =~ /<time/) {
      my %tags = parse_sgml_tag $line;
      my $new_time = $tags{'sec'};
      if (@text > 0) {
        print "$sgml_file $filename $turn_speaker $turn_dialect $turn_sex $time $new_time ";
        print join(" ", @text) . "\n";
      }
      @text = ();
      $time = $new_time;
      ;
    } elsif ($line =~ /<\/time/) {
      #print $line;
      ;
    } elsif ($line =~ /<foreign/) {
      $line = trim $line;
      push @text, $line;
    } elsif ($line =~ /<\/foreign/) {
      $line = trim $line;
      push @text, $line;
      ;
    } elsif ($line =~ /<unclear/) {
      $line = trim $line;
      push @text, $line;
    } elsif ($line =~ /<\/unclear/) {
      $line = trim $line;
      push @text, $line;
      ;
    } elsif ($line =~ /<[^\/]/) {
      parse_sgml_tag $line;
      print STDERR "$0: INFO: Unknown tag $line in file $file\n";
    } elsif ($line =~ /<\//) {
      ;
    } else {
      $line = trim $line;
      push @text, $line if $line;
      ;
    }

  }
  close($f);
}
