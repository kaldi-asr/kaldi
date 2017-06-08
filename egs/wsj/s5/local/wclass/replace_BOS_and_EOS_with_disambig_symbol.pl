#!/usr/bin/perl -w

# Copyright 2016 FAU Erlangen (Author: Axel Horndasch)
# Apache 2.0.
#
use strict;
use warnings;
use Getopt::Long;

my $Usage = <<EOU;
Usage: replace_BOS_and_EOS_with_disambig_symbol.pl [options] < in.fst > out.fst

This script replaces <s>:<s> (beginning of sentence / BOS) and </s>:</s> (end
of sentence / EOS) of an FST with the symbols provided by the command-line
options. Default values are "#0:<eps>" (both for BOS and EOS.

Example: <s>:<s> -> #CITYNAME:<eps> and </s>:</s> -> #CITYNAME:<eps>

If the options 'remove_bos_weight' (or 'remove_eos_weight') are used, the
weight(s) of the <s>:<s> (or </s>:</s>) labelled transitions are removed.

Allowed options:
   --bos-input-symbol          : Replace input  <s>  with this symbol          (string,  default = "#0")
   --bos-output-symbol         : Replace output <s>  with this symbol          (string,  default = "<eps>")
   --eos-input-symbol          : Replace input  </s> with this symbol          (string,  default = "#0")
   --eos-output-symbol         : Replace output </s> with this symbol          (string,  default = "<eps>")
   --remove-bos-weight         : Remove weight of <s>:<s> transition           (string,  default = "false")
   --remove-eos-weight         : Remove weight of </s>:</s> transition         (string,  default = "false")

EOU

# command line options
my $bos_input_symbol  = "#0";
my $bos_output_symbol = "<eps>";
my $eos_input_symbol  = "#0";
my $eos_output_symbol = "<eps>";

my $remove_bos_weight = "false";
my $remove_eos_weight = "false";

# get the optional command line options
GetOptions(
    "bos-input-symbol=s"  => \$bos_input_symbol,
    "bos-output-symbol=s" => \$bos_output_symbol,
    "eos-input-symbol=s"  => \$eos_input_symbol,
    "eos-output-symbol=s" => \$eos_output_symbol,

    "remove-bos-weight=s" => \$remove_bos_weight,
    "remove-eos-weight=s" => \$remove_eos_weight,
    ) or die "$Usage";

($remove_bos_weight eq "true" || $remove_bos_weight eq "false") ||
  die "$0: Bad value for option --remove-bos-weight \"$remove_bos_weight\"\n";
($remove_eos_weight eq "true" || $remove_eos_weight eq "false") ||
  die "$0: Bad value for option --remove-eos-weight \"$remove_eos_weight\"\n";

my $states;
my $old_input_symbol;
my $space;
my $old_output_symbol;
my $weight;

# Go through the FST transition by transition (== line by line)
while (<STDIN>){
  # We are looking for transitions wth BOS or EOS input/output symbols
  if (/^(\d+\s+\d+\s+)(\<\/?s\>)(\s+)(\<\/?s\>)(.*)?$/) {
    $states            = $1;
    $old_input_symbol  = $2;
    $space             = $3;
    $old_output_symbol = $4;
    if (defined($5)) {
      $weight = $5;
    } else {
      undef($weight);
    }
    # BOS (beginning of sentence)
    if ($old_input_symbol eq "<s>" and $old_output_symbol eq "<s>") {
      print "$states$bos_input_symbol$space$bos_output_symbol";
      if (($remove_bos_weight eq "false") and defined($weight)) {
	print "$weight";
      }
    } elsif ($old_input_symbol eq "</s>" and $old_output_symbol eq "</s>") {
      print "$states$eos_input_symbol$space$eos_output_symbol";
      if (($remove_eos_weight eq "false") and defined($weight)) {
	print "$weight";
      }
    } else {
      warn "$0: Oops, something is messed up with <s> and </s> symbols";
    }
    print "\n";
  } else {
    print;
  }
}

