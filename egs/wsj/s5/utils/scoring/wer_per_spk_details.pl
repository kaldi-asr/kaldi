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
use List::Util qw[max];
use Getopt::Long;
use Pod::Usage;


#use Data::Dumper;

my $WIDTH=10;
my $SPK_WIDTH=15;
my $help;

GetOptions("spk-field-width" => \$SPK_WIDTH,
           "field-width" => \$WIDTH,
           "help|?" => \$help
           ) or pod2usage(2);
pod2usage(1) if $help;
pod2usage("$0: Too many files given.\n")  if (@ARGV != 1);

my %UTTMAP;
my %PERSPK_STATS;

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

sub print_header {
  
  my $f="%${WIDTH}s";
  my $str = sprintf("%-${SPK_WIDTH}s id  $f $f $f $f $f $f $f $f\n", "SPEAKER", 
                    "#SENT", "#WORD", "Corr", "Sub", "Ins", "Del", "Err", "S.Err");
  return $str;
}
sub format_raw {
  my $spk = $_[0];
  my $sent = $_[1];
  my $word = $_[2];
  my $c = $_[3];
  my $s = $_[4];
  my $i = $_[5];
  my $d = $_[6];
  my $err = $_[7];
  my $serr = $_[8];

  my $f = "%${WIDTH}d"; 
  my $str = sprintf("%-${SPK_WIDTH}s raw $f $f $f $f $f $f $f $f\n", $spk, 
                    $sent, $word, $c, $s, $i, $d, $err, $serr);
  return $str;
}
sub format_sys {
  my $spk = $_[0];
  my $sent = $_[1];
  my $word = $_[2];
  my $c = $_[3];
  my $s = $_[4];
  my $i = $_[5];
  my $d = $_[6];
  my $err = $_[7];
  my $serr = $_[8];

  my $fd = "%${WIDTH}d"; 
  my $ff = "%${WIDTH}.2f"; 
  my $str = sprintf("%-${SPK_WIDTH}s sys $fd $fd $ff $ff $ff $ff $ff $ff\n", $spk, 
                    $sent, $word, $c, $s, $i, $d, $err, $serr);
  return $str;
}

open(UTT2SPK,$ARGV[0]) or die "Could not open the utt2spk file $ARGV[0]";
while(<UTT2SPK>) {
  chomp;
  my @F=split;
  die "Incompatible format of the utt2spk file: $_" if @F != 2; 
  $UTTMAP{$F[0]} = $F[1];
  # Set width of speaker column by its longest label,
  if($SPK_WIDTH < length($F[1])) { $SPK_WIDTH = length($F[1]) }
}
close(UTT2SPK);

(my $is_utf8, my @text) = get_utf8_or_bytestream(\*STDIN);
if ($is_utf8) {
  binmode(STDOUT, ":utf8");
}

while (@text) {
  my $line = shift @text;
  chomp $line;
  my @entries = split(" ", $line);
  next if  @entries < 2;
  next if  $entries[1] ne "#csid" ; 
  die "Incompatible entry $_ " if @entries != 6;

  my $c=$entries[2]; 
  my $s=$entries[3]; 
  my $i=$entries[4]; 
  my $d=$entries[5]; 
  
  my $UTT=$entries[0];
  my $SPK=$UTTMAP{$UTT};
  $PERSPK_STATS{$SPK}->{"C"} += $c;
  $PERSPK_STATS{$SPK}->{"S"} += $s;
  $PERSPK_STATS{$SPK}->{"I"} += $i;
  $PERSPK_STATS{$SPK}->{"D"} += $d;
  $PERSPK_STATS{$SPK}->{"SENT"} += 1;
  $PERSPK_STATS{$SPK}->{"SERR"} += 1 if ($s + $i + $d != 0);
}

my $C = 0;
my $S = 0;
my $I = 0;
my $D = 0;
my $SENT = 0;
my $WORD = 0;
my $ERR = 0;
my $SERR = 0;

print print_header;

for my $SPK (sort (keys %PERSPK_STATS)) {
  my $c=$PERSPK_STATS{$SPK}->{"C"}; 
  my $s=$PERSPK_STATS{$SPK}->{"S"}; 
  my $i=$PERSPK_STATS{$SPK}->{"I"}; 
  my $d=$PERSPK_STATS{$SPK}->{"D"}; 
  my $sent=$PERSPK_STATS{$SPK}->{"SENT"} ;
  my $word=$c+$s+$d;
  my $err =$s+$d+$i;
  my $serr = $PERSPK_STATS{$SPK}->{"SERR"} // 0;

  my $spk = "$SPK";
  $C += $c; $S += $s; $I += $i; $D += $d; 
  $SENT += $sent; $SERR += $serr;

  my $w = 1.0 *$word;
  print format_raw($spk, $sent, $word, $c, $s, $i, $d, $err, $serr);
  print format_sys($spk, $sent, $word, 100 * $c/$w, 100 * $s/$w, 
                   100 * $i/$w, 100 * $d/$w, 100 * $err/$w, 100.0 * $serr/$sent) unless $w == 0;

}
$WORD= $C + $S + $D;
$ERR= $S + $D + $I;
my $W = 1.0 * $WORD;

print format_raw("SUM", $SENT, $WORD, $C, $S, $I, $D, $ERR, $SERR);
print format_sys("SUM", $SENT, $WORD, 100* $C/$W, 100*$S/$W, 
                         100*$I/$W,100*$D/$W,100*$ERR/$W, 100.0 * $SERR/$SENT) unless $W==0;


 __END__

=head1 NAME
  wer_per_spk_details.pl -- generate aggregated per-speaker details

=head1 SYNOPSIS

  wer_per_spk_details.pl  data/dev/utt2spk

  Options:
    --spk-field-width         Width of the first field (spk ID field)
    --field-width             Width of the fields (with exception of the SPK ID 
                              field)

=head1 DESCRIPTION
  This program aggregates the per-utterance output from utils/wer_per_utt_details.pl
  It cares only about the "#csid" field (counts of Corr, Sub, Ins and Del);

  It expects one parameter -- file in the format of the kaldi utt2spk.
  In case the SPK ID is longer that 15 characters, the parameter spk-field-width
  can be used; the same for all other fields and field-width parameter.
  The field-width parameter should not be necessary under normal circumstances.

==head1 EXAMPLE INPUT AND OUTPUT
  Input:
    UTT-A #csid 3 1 1 1

  Output:
    SPEAKER         id       #SENT      #WORD       Corr        Sub        Ins        Del        Err      S.Err
    A               raw          1          5          3          1          1          1          3          1
    A               sys          1          5      60.00      20.00      20.00      20.00      60.00     100.00
    SUM             raw          1          5          3          1          1          1          3          1
    SUM             sys          1          5      60.00      20.00      20.00      20.00      60.00     100.00
    
    The input can contain other lines as well -- those will be ignored during
    reading the input. I.E. this is a completely legal input:
      
      UTT-A ref  word-A   <eps>  word-B  word-C  word-D  word-E
      UTT-A hyp  word-A  word-A  word-B   <eps>  word-D  word-X
      UTT-A op      C       I       C       D       C       S
      UTT-A #csid 3 1 1 1

=cut
