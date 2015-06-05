#!/usr/bin/env perl

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)
# Apache 2.0.
#

use strict;
use warnings;
use Getopt::Long;

my $Usage = <<EOU;
Usage: annotated_kwlist_to_KWs.pl [options] <kwlist.annot.xml|-> <keywords|-> [category]
 e.g.: annotated_kwlist_to_KWs.pl kwlist.annot.list keywords.list "NGram Order:2,3,4"

This script reads an annotated kwlist xml file and writes a list of keywords, according
to the given categories. The "category" is a "key:value" pair in the annotated kwlist xml
file. For example
1. "NGram Order:2,3,4"
2. "NGram Order:2"
3. "NGram Order:-"
where "NGram Order" is the category name. The first line means print keywords that are
bigram, trigram and 4gram; The second line means print keywords only for bigram; The last
line means print all possible ngram keywords.
If no "category" is specified, the script will print out the possible categories.

Allowed options:
EOU

GetOptions(); 

@ARGV >= 2 || die $Usage;

# Workout the input/output source
my $kwlist_filename = shift @ARGV;
my $kws_filename = shift @ARGV;

my $source = "STDIN";
if ($kwlist_filename ne "-") {
  open(KWLIST, "<$kwlist_filename") || die "Fail to open kwlist file: $kwlist_filename\n";
  $source = "KWLIST";
}

# Process kwlist.annot.xml
my %attr;
my %attr_kws;
my $kwid="";
my $name="";
my $value="";
while (<$source>) {
  chomp;
  if (m/<kw kwid=/) {($kwid) = /kwid="(\S+)"/; next;}
  if (m/<name>/) {($name) = /<name>(.*)<\/name>/; next;}
  if (m/<value>/) {
    ($value) = /<value>(.*)<\/value>/;
    if (defined($attr{$name})) {
      $attr{"$name"}->{"$value"} = 1;
    } else {
      $attr{"$name"} = {"$value", 1};
    }
    if (defined($attr_kws{"${name}_$value"})) {
      $attr_kws{"${name}_$value"}->{"$kwid"} = 1;
    } else {
      $attr_kws{"${name}_$value"} = {"$kwid", 1};
    }
  }
}

my $output = "";
if (@ARGV == 0) {
  # If no category provided, print out the possible categories
  $output .= "Possible categories are:\n\n";
  foreach my $name (keys %attr) {
    $output .= "$name:";
    my $count = 0;
    foreach my $value (keys %{$attr{$name}}) {
      if ($value eq "") {$value = "\"\"";}
      if ($count == 0) {
        $output .= "$value";
        $count ++; next;
      } 
      if ($count == 6) {
        $output .= ", ...";
        last;
      }
      $output .= ",$value"; $count ++;
    }
    $output .= "\n";
  }
  print STDERR $output;
  $output = "";
} else {
  my %keywords;
  while (@ARGV > 0) {
    my $category = shift @ARGV;
    my @col = split(/:/, $category);
    @col == 2 || die "Bad category \"$category\"\n";
    $name = $col[0];
    if ($col[1] eq "-") {
      foreach my $value (keys %{$attr{$name}}) {
        foreach my $kw (keys %{$attr_kws{"${name}_$value"}}) {
          $keywords{$kw} = 1;
        }
      }
    } else {
      my @col1 = split(/,/, $col[1]);
      foreach my $value (@col1) {
        foreach my $kw (keys %{$attr_kws{"${name}_$value"}}) {
          $keywords{$kw} = 1;
        }
      }
    }
  }
  foreach my $kw (keys %keywords) {
    $output .= "$kw\n";
  }
}

if ($kwlist_filename ne "-") {close(KWLIST);}
if ($kws_filename eq "-") { print $output;}
else {
  open(O, ">$kws_filename") || die "Fail to open file $kws_filename\n";
  print O $output;
  close(O);
}
