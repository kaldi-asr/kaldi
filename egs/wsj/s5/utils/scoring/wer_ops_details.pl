#!/usr/bin/env perl
# Copyright 2015 Johns Hopkins University (Author: Yenda Trmal <jtrmal@gmail.com>)

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


# These scripts are (or can be) used by scoring scripts to generate 
# additional information (such as per-spk wer, per-sentence alignments and so on) 
# during the scoring. See the wsj/local/score.sh script for example how 
# the scripts are used
# For help and instructions about usage, see the bottom of this file, 
# or call it with the parameter --help
 
use strict;
use warnings;
use utf8;
#use List::Util qw[max];
use Getopt::Long;
use Pod::Usage;


binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";

my $help;
my $special_symbol= "<eps>";
my $separator=";";

GetOptions("special-symbol=s" => \$special_symbol,
           "separator=s" => \$separator,
           "help|?" => \$help
           ) or pod2usage(2);
pod2usage(1) if $help;
pod2usage("$0: Too many files given.\n")  if (@ARGV != 0);

my %EDIT_OPS;
my %UTT;
while (<STDIN>) {
  chomp;
  my @entries = split(" ", $_);
  next if  @entries < 2;
  next if  ($entries[1] ne "hyp") and ($entries[1] ne "ref") ; 
  if (scalar @entries <= 2 ) {
    print STDERR "Warning: skipping entry \"$_\", either an  empty phrase or incompatible format\n" ;
    next;
  }

  die "The input stream contains duplicate entry $entries[0] $entries[1]\n" 
    if exists $UTT{$entries[0]}->{$entries[1]};
  push @{$UTT{$entries[0]}->{$entries[1]}}, @entries[2..$#entries];
  #print join(" ", @{$UTT{$entries[0]}->{$entries[1]}}) . "\n";
  #print $_ . "\n";
}

for my $utterance( sort (keys %UTT) ) {
  
  die "The input stream does not contain entry \"hyp\" for utterance $utterance\n" 
    unless exists $UTT{$utterance}->{"hyp"};
  die "The input stream does not contain entry \"ref\" for utterance $utterance\n" 
    unless exists $UTT{$utterance}->{"ref"};

  my $hyp = $UTT{$utterance}->{"hyp"};
  my $ref = $UTT{$utterance}->{"ref"};

  die "The \"ref\" an \"hyp\" entries do not have the same number of fields"
    unless (scalar @{$hyp}) == (scalar @{$ref});

  for ( my $i = 0; $i < @{$hyp}; $i += 1) {
    $EDIT_OPS{$ref->[$i]}->{$hyp->[$i]} += 1;
  }
}

foreach my $refw ( sort (keys %EDIT_OPS) ) {
  foreach my $hypw ( sort (keys %{$EDIT_OPS{$refw}} ) ) {
    if ( $refw eq $hypw ) {
      print "correct $refw $hypw " . $EDIT_OPS{$refw}->{$hypw} . "\n";
    } elsif ( $refw eq $special_symbol ) {
      print "insertion $refw $hypw " . $EDIT_OPS{$refw}->{$hypw} . "\n";
    } elsif ( $hypw eq $special_symbol ) {
      print "deletion $refw $hypw " . $EDIT_OPS{$refw}->{$hypw} . "\n";
    } else {
      print "substitution $refw $hypw " . $EDIT_OPS{$refw}->{$hypw} . "\n";
    }
  }
}
exit 0;
__END__
=head1 NAME
  wer_ops_details.pl -- generate aggregated ops statistics

=head1 SYNOPSIS

  wer_per_spk_details.pl 
  
  Options:
    --special-symbol        special symbol used in align-text to denote empty word 
                            in case insertion or deletion ("<eps>" by default)
    --help                  Print this help

==head1 DESCRIPTION
  The program generates global statistic on how many time was each word 
  recognized correctly, confused as another word, incorrectly deleted or inserted.
  The output will contain similar info as the sclite dtl file, the format is,
  however, completely different.



==head1 EXAMPLE INPUT AND OUTPUT
  Input:
    UTT-A ref  word-A   <eps>  word-B  word-C  word-D  word-E
    UTT-A hyp  word-A  word-A  word-B   <eps>  word-D  word-X

  Output:
    correct       word-A  word-A  1
    correct       word-B  word-B  1
    correct       word-D  word-D  1
    deletion      word-C  <eps>   1
    insertion     <eps>   word-A  1
    substitution  word-E  word-X  1


  Note:
    The input can contain other lines as well -- those will be ignored during
    reading the input. I.E. this is a completely legal input:
      
      UTT-A ref  word-A   <eps>  word-B  word-C  word-D  word-E
      UTT-A hyp  word-A  word-A  word-B   <eps>  word-D  word-X
      UTT-A op      C       I       C       D       C       S
      UTT-A #csid 3 1 1 1
=cut
