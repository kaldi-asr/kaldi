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
use Getopt::Long;
use Pod::Usage;


my $help;
my $special_symbol= "<eps>";
my $separator=";";
my $extra_size=4;
my $max_size=16;

# this function reads the opened file (supplied as a first
# parameter) into an array of lines. For each
# line, it tests whether it's a valid utf-8 compatible
# line. If all lines are valid utf-8, it returns the lines 
# decoded as utf-8, otherwise it assumes the file's encoding
# is one of those 1-byte encodings, such as ISO-8859-x
# or Windows CP-X.
# Please recall we do not really care about
# the actually encoding, we just need to 
# make sure the length of the (decoded) string 
# is correct (to make the output formatting looking right).
sub get_utf8_or_bytestream {
  use Encode qw(decode encode);
  my $is_utf_compatible = 1;
  my @unicode_lines;
  my @raw_lines;
  my $raw_text;
  my $lineno = 0;
  my $file = shift;

  while (<$file>) {
    $raw_text = $_;
    last unless $raw_text;
    if ($is_utf_compatible) {
      my $decoded_text = eval { decode("UTF-8", $raw_text, Encode::FB_CROAK) } ;
      $is_utf_compatible = $is_utf_compatible && defined($decoded_text); 
      push @unicode_lines, $decoded_text;
    }
    push @raw_lines, $raw_text;
    $lineno += 1;
  }

  if (!$is_utf_compatible) {
    print STDERR "$0: Note: handling as byte stream\n";
    return (0, @raw_lines);
  } else {
    print STDERR "$0: Note: handling as utf-8 text\n";
    return (1, @unicode_lines);
  }

  return 0;
}
sub print_line {
  my $op = $_[0];
  my $rewf = $_[1];
  my $hypw = $_[2];
  my $nofop = $_[3];

}

sub max {
  $_[ 0 ] < $_[ -1 ] ? shift : pop while @_ > 1;
  return @_;
}


GetOptions("special-symbol=s" => \$special_symbol,
           "separator=s" => \$separator,
           "help|?" => \$help
           ) or pod2usage(2);
pod2usage(1) if $help;
pod2usage("$0: Too many files given.\n")  if (@ARGV != 0);

my %EDIT_OPS;
my %UTT;
(my $is_utf8, my @text) = get_utf8_or_bytestream(\*STDIN);
if ($is_utf8) {
  binmode(STDOUT, ":utf8");
}

while (@text) {
  my $line = shift @text;
  chomp $line;
  my @entries = split(" ", $line);
  next if  @entries < 2;
  next if  ($entries[1] ne "hyp") and ($entries[1] ne "ref") ;
  if (scalar @entries <= 2 ) {
    print STDERR "$0: Warning: skipping entry \"$_\", either an  empty phrase or incompatible format\n" ;
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

my $word_len = 0;
my $ops_len =0;
foreach my $refw ( sort (keys %EDIT_OPS) ) {
  foreach my $hypw ( sort (keys %{$EDIT_OPS{$refw}} ) ) {
    my $q = length($refw) > length($hypw) ? length($refw):  length($hypw) ;
    if ( $q > $max_size ) {
      #print STDERR Dumper( [$refw, $hypw, $q, length($refw), length($hypw) ]);
      ;
    }
    $word_len = $q > $word_len ? $q : $word_len ;

    my $d = length(sprintf("%d", $EDIT_OPS{$refw}->{$hypw}));
    $ops_len =  $d > $ops_len ? $d: $ops_len ;
  }
}

if ($word_len > $max_size) {
  ## We used to warn about this, but it was just confusing-- dan.
  ## print STDERR "wer_ops_details.pl [info; affects only whitespace]: we are limiting the width to $max_size, max word len was $word_len\n";
  $word_len = $max_size
};


foreach my $refw ( sort (keys %EDIT_OPS) ) {
  foreach my $hypw ( sort (keys %{$EDIT_OPS{$refw}} ) ) {
    if ( $refw eq $hypw ) {
      printf "correct       %${word_len}s    %${word_len}s    %${ops_len}d\n", ($refw,  $hypw,  $EDIT_OPS{$refw}->{$hypw});
    } elsif ( $refw eq   $special_symbol ) {
      printf "insertion     %${word_len}s    %${word_len}s    %${ops_len}d\n", ($refw,  $hypw,  $EDIT_OPS{$refw}->{$hypw});
    } elsif ( $hypw eq $special_symbol ) {
      printf "deletion      %${word_len}s    %${word_len}s    %${ops_len}d\n", ($refw,  $hypw,  $EDIT_OPS{$refw}->{$hypw});
    } else {
      printf "substitution  %${word_len}s    %${word_len}s    %${ops_len}d\n", ($refw,  $hypw,  $EDIT_OPS{$refw}->{$hypw});
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
