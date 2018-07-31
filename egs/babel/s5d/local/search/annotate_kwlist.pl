#!/usr/bin/env perl
#===============================================================================
# Copyright 2016  (Author: Yenda Trmal <jtrmal@gmail.com>)
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

my $Usage = <<EOU;
A simple script for the F4DE kwlist format annotation (adding info fields
which can be used for analysis of the results. The script reads the xml file
from it's STDIN (encoding will be autodetected) and prints utf-8 encoded
kwlist xml file to STDOUT. The annotations will be generated from the category
table (used in the native kaldi-kws pipeline

Usage: $0 [options] <category-table>  > output.kwlist.xml
 e.g.: cat kwlist.xml | $0 data/dev10h.pem/kwset_kwlist/categories > output.kwlist.xml

EOU
use strict;
use warnings "FATAL";
use utf8;
use XML::Parser;
use Data::Dumper;

binmode STDERR, ":utf8";
binmode STDOUT, ":utf8";

my $IN_KWTEXT=0;
my $KWTEXT='';
my $KWID='';
my %CATEGORIES;

sub kwlist {
  my @entries =  @_;
  shift @entries;
  shift @entries;

  my $header="";
  while (@entries) {
    my $k = shift @entries;
    my $w = shift @entries;

    $header .= " $k=\"$w\" ";
  }
  print "<kwlist $header>\n";
}

sub kwlist_ {
  print "</kwlist>\n";
}

sub kw {
  my @entries =  @_;
  shift @entries;
  shift @entries;
  #print Dumper(@entries);
  my %params = @entries;
  $KWID = $params{kwid};
}

sub kwtext {
  my @entries =  @_;
  shift @entries;
  $IN_KWTEXT=1;
  #print Dumper(@entries);
}
sub char {
  my @entries =  @_;
  shift @entries;
  $KWTEXT=$entries[0] if $IN_KWTEXT eq 1;
}

sub kwtext_ {
  my @entries =  @_;
  shift @entries;
  $IN_KWTEXT=0;
  if ($KWTEXT) {
    if (exists $CATEGORIES{$KWID}) {
      print "  <kw kwid=\"$KWID\">\n";
      print "    <kwtext>$KWTEXT</kwtext>\n";
      print "    <kwinfo>\n";
      print "      <attr>\n";
      print "        <name>ALL</name>\n";
      print "        <value>1</value>\n";
      print "      </attr>\n";
      foreach my $cat (sort keys %{$CATEGORIES{$KWID}} ) {
        my @entries = split("=", $cat);
        my $name;
        my $value;

        if (scalar @entries == 2) {
          $name = $entries[0];
          $value = $entries[1];
        } else {
          $name = $cat;
          $value = 1;
        }
        print "      <attr>\n";
        print "        <name>$name</name>\n";
        print "        <value>$value</value>\n";
        print "      </attr>\n";
      }
      print "    </kwinfo>\n";
      print "  </kw>\n";
    } else {
      my $n = scalar split " ", $KWTEXT;
      my $l=length join("", split($KWTEXT));

      $n = sprintf "%02d", $n;
      $l = sprintf "%02d", $l;

      print "  <kw kwid=\"$KWID\">\n";
      print "    <kwtext>$KWTEXT</kwtext>\n";
      print "    <kwinfo>\n";
      print "      <attr>\n";
      print "        <name>Characters</name>\n";
      print "        <value>$l</value>\n";
      print "      </attr>\n";
      print "      <attr>\n";
      print "        <name>NGramOrder</name>\n";
      print "        <value>$n</value>\n";
      print "      </attr>\n";
      print "      <attr>\n";
      print "        <name>NGram Order</name>\n";
      print "        <value>$n</value>\n";
      print "      </attr>\n";
      print "    </kwinfo>\n";
      print "  </kw>\n";
    }
  }
}

if (@ARGV != 1) {
  print STDERR "Incorrect number of command-line parameters\n";
  print STDERR "$Usage\n";
  die "Cannot continue\n"
}


#Read the categories table
open(G, $ARGV[0]) or die "Cannot open the categories table $ARGV[0]";
while (my $line = <G>) {
  my @entries = split(" ", $line);
  my $kwid = shift @entries;

  foreach my $group (@entries) {
    $CATEGORIES{$kwid}->{$group} = 1;
  }
}
close(G);

my $p1 = new XML::Parser(Style => 'Subs');
$p1->setHandlers(Char  => \&char);
$p1->parse(*STDIN);

