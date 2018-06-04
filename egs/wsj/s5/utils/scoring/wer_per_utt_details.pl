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


#These scripts are (or can be) used by scoring scripts to generate 
#additional information (such as per-spk wer, per-sentence alignments and so on) 
#during the scoring. See the wsj/local/score.sh script for example how 
#the scripts are used
#For help and instructions about usage, see the bottom of this file, 
#or call it with the parameter --help
#
use strict;
use warnings;
use List::Util qw[max];
use Getopt::Long;
use Pod::Usage;


#use Data::Dumper;

my $special_symbol= "<eps>";
my $separator=";";
my $output_hyp = 1;
my $output_ref = 1;
my $output_ops = 1;
my $output_csid = 1;
my $help;

GetOptions("special-symbol=s" => \$special_symbol,
           "separator=s" => \$separator,
           "output-hyp!" => \$output_hyp,
           "output-ref!" => \$output_ref,
           "output-ops!" => \$output_ops,
           "output-csid!" => \$output_csid,
           "help|?" => \$help
           ) or pod2usage(2);
pod2usage(1) if $help;
pod2usage("$0: Too many parameters.\n")  if (@ARGV != 0);

sub rjustify {
  my $maxlen =  $_[1];
  my $str =  $_[0];
  return sprintf("%-${maxlen}s", $str);
}
sub ljustify {
  my $maxlen =  $_[1];
  my $str =  $_[0];
  return sprintf("%${maxlen}s", $str);
}
sub cjustify {
  my $maxlen =  $_[1];
  my $str =  $_[0];
  my $right_spaces = int(($maxlen - length($str)) / 2);
  my $left_spaces =$maxlen - length($str) - $right_spaces;
  return sprintf("%s%s%s", " " x $left_spaces,  $str, " " x $right_spaces);
}

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
}

(my $is_utf8, my @text) = get_utf8_or_bytestream(\*STDIN);
if ($is_utf8) {
  binmode(STDOUT, ":utf8");
}

while (@text) {
  my $line = shift @text;
  chomp $line;
  (my $utt_id, my $alignment) = split (" ", $line, 2);
  my @alignment_pairs = split(" ", $alignment); #splits on spaces, does not create empty fields
 
  my @HYP;
  my @REF;
  my @OP;
  my %OPCOUNTS= (
    "I" => 0,
    "D" => 0,
    "S" => 0,
    "C" => 0
  );

  while(@alignment_pairs) {
    my $ref = shift @alignment_pairs;
    my $hyp = shift @alignment_pairs;
    if (@alignment_pairs) {
      my $sep = shift @alignment_pairs;
      die "Detected incorrect separator $sep (expected $separator).\n" unless ($sep eq $separator);
    }

    push @HYP, $hyp;
    push @REF, $ref;

    if ( $hyp eq $special_symbol ) {
      push @OP, "D";
      $OPCOUNTS{"D"} +=1;
    } elsif ( $ref eq $special_symbol ) {
      push @OP, "I";
      $OPCOUNTS{"I"} +=1;
    } elsif ($ref ne $hyp ) {
      push @OP, "S";
      $OPCOUNTS{"S"} +=1;
    } else {
      push @OP, "C";
      $OPCOUNTS{"C"} +=1;
    }
  }

  die "Number of edit ops is not equal to the length of the text for utterance $utt_id\n" if scalar(@OP) != scalar(@HYP);
   
  my @hyp_str;
  my @ref_str;
  my @op_str;
  for (my $i=0; $i <= $#OP; $i+=1) {
    my $maxlen=max(length($REF[$i]), length($HYP[$i]), length($OP[$i]));

    push @ref_str, cjustify($REF[$i], $maxlen);
    push @hyp_str, cjustify($HYP[$i], $maxlen);
    push @op_str, cjustify($OP[$i], ${maxlen});
  }
  print $utt_id . " ref  " . join("  ", @ref_str) . "\n" if $output_ref;
  print $utt_id . " hyp  " . join("  ", @hyp_str) . "\n" if $output_hyp;
  print $utt_id . " op   " . join("  ", @op_str) . "\n" if $output_ops;
  print $utt_id . " #csid" . " " .$OPCOUNTS{"C"} . " " . $OPCOUNTS{"S"} . " " . $OPCOUNTS{"I"} . " " . $OPCOUNTS{"D"} . "\n" if $output_csid;
}


 __END__

=head1 NAME
  wer_per_utt_details.pl -- generate detailed stats

=head1 SYNOPSIS

  Example:
    align-text ark:text.filt ark:10.txt ark,t:-  | wer_per_utt_details.pl

  Options:
    --special-symbol        special symbol used in align-text to denote empty word 
                            in case insertion or deletion ("<eps>" by default)
    --separator             special symbol used to separate individual word-pairs
                            in the align-text output (";" by default)

    --[no]output-hyp        disable/enable printing of the hyp (hypothesis) entry
    --[no]output-ref        disable/enable printing of the ref (reference) entry
    --[no]output-ops        disable/enable printing of the ops (edit operations) entry
    --[no]output-csid       disable/enable printing of the #csid entry (counts
                            of the individual edit operations)

=head1 DESCRIPTION
    The program works as a filter -- reads the output from align-text program,
    parses it and outputs the requested entries on the output. The format of
    the entries was chosen so that it allows for easy parsing while being human
    readable.

    By default, all entries (hyp, ref, ops, #csid) are printed. 

    The filter can be used (for example) to generate detailed statistics
    from scoring (similar to the dtl/prf output of the sctk sclite outut)

==head1 EXAMPLE INPUT AND OUTPUT
  Input:
    "UTT-A word-A word-A; <eps> word-A; word-B word-B; word-C <eps>; word-D word-D; word-E word-X;

  Output:
    UTT-A ref  word-A   <eps>  word-B  word-C  word-D  word-E
    UTT-A hyp  word-A  word-A  word-B   <eps>  word-D  word-X
    UTT-A op      C       I       C       D       C       S
    UTT-A #csid 3 1 1 1

=cut

