#!/usr/bin/env perl

#===============================================================================
# Copyright (c) 2019  Johns Hopkins University (Author: Jinyi Yang)
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
use Encode;
use Time::Piece;

require HTML::Parser or die "This script needs HTML::Parser from CPAN";
HTML::Parser->import();

binmode(STDOUT, ":utf8");

sub  trim { my $s = shift; $s =~ s/^\s+|\s+$//g; return $s };

sub get_doc_no {
  my $tag = shift(@_);
  my @tmpdoc = split /\s+/, $tag;
  my @doc_nos = split /\./, $tmpdoc[1];
  return @doc_nos;
}

sub check_doc_type {
  my $tag = shift(@_);
  if ( $tag =~ /UNTRANSCRIBED/){
    return 0;
    ;
  } else {
      return 1;
  }
}

sub str2time {
  my ($str) = @_;
  $str =~ s/(\.[0-9]+)?\z//;
  my $fraction = $1 || 0;
  return Time::Piece->strptime($str, '%H:%M:%S')->epoch + $fraction;
}

sub get_time_tag{
    my $start = shift(@_);
    my $end = shift(@_);
    if (($start ne "") && ($end ne "")) {
      $start = sprintf("%.2f", $start);
      $end = sprintf("%.2f", $end);
      my $tag = sprintf("%06.0f_%06.0f", 100*$start+0.5, 100*$end+0.5);
     return $tag;
     ;
    } else{
       print STDERR "$0: Empty time tag: $start or $end\n";
       return "";
    }
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
  my $filename = "";
  my $docname = "";
  my $doctype = "";
  my @docno = ();
  my $doc_id = "";
  my @text = ();
  my $start_time = 0;
  my $end_time = 0;
  my $doc_start_time = 0;
  my $current_time = 0;
  my @times = ();

  my $sgml_file = `basename $file`;
  $sgml_file = trim $sgml_file;
  $sgml_file =~ s/\.src_sgm$//g;
  my @sgml_file_ids = split '_', $sgml_file;
  my $sgml_file_id = $sgml_file_ids[3].$sgml_file_ids[0].$sgml_file_ids[1];

  open(my $f, '<:encoding(iso-8859-1)', $file) or die "Could not open file $file: $?\n";
  while(my $line = <$f>) {
    $line = trim $line;
    next unless $line;

    if ($line =~ /<DOCNO>/) {
      @docno = get_doc_no $line;
      $doc_id = $docno[0].$docno[1];
      $doc_id = $docno[2]; # Four digits
      ;
    } elsif($line =~ /<DATE_TIME>/ ){
        @times = split /\s+/, $line;
        $current_time = str2time($times[2]);
        if ($doc_start_time ==  0){
          $doc_start_time = $current_time;
          $start_time = 0;
          ;
        } else {
          $start_time = $current_time - $doc_start_time;
        }
      ;
    } elsif ($line =~ /<TURN>/){
        $line = "THISISSPKTURN"; # Replace <TURN> with a word, indicating speaker change, will be removed from text before LM training
        push @text, $line;
      ;
    } elsif($line =~ /<END_TIME>/){
        @times = split /\s+/, $line;
        $end_time = str2time($times[2]) - $doc_start_time;
      ;
    } elsif ($line =~ /<DOCTYPE>/) {
      $doctype = check_doc_type $line;
      ;
    } elsif ($line eq "<\/DOC>") {
      if ((@text > 0) && ($doctype)) {
        if ($end_time <= $start_time){
          print STDERR "$0: WARNING: File $file has invalid time tag at <DOCNO> $doc_id\n";
        }
        my $time_tag = get_time_tag($start_time, $end_time);
        $docname = $sgml_file."_".$doc_id."_".$time_tag;
        print "$docname ";
        print join(" ", @text) . "\n";
      }
      $docname = "";
      @text = ();
      ;
    } elsif ($line !~ "<") {
      $line = trim $line;
      $line = decode("gbk", $line);
      $line =~ s:〈turn〉:THISISSPKTURN:g;
      $line =~ s:<turn>:THISISSPKTURN:g;
      $line =~ s:turn>:THISISSPKTURN:g;
      $line =~ s:<turn:THISISSPKTURN:g;
      push @text, $line if $line;
      ;
    }
  }
  close($f);
}
