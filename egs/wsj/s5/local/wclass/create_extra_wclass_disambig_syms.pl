#!/usr/bin/perl -w
# Copyright 2016 FAU Erlangen (Author: Axel Horndasch)
# Apache 2.0.
#
use strict;
use warnings;
use Getopt::Long;

my $Usage = <<EOU;
Usage:  create_extra_wclass_disambig_syms.pl [options] wclass_list.txt > extra_wclass_disambig_syms.txt

This script creates a list of extra (word) disambiguation symbols which are
needed for words.txt and other files. The list can be attached to words.txt and
the other files by running prepare_lang.sh with the option
"--extra-word-disambig-syms".

The input file (e.g. wclass_list.txt) is assumed to have the following format
CLASS_NAME_1<TAB>0.2
CLASS_NAME_2<TAB>0.0
...
The second value is a a probability-like value for OOV detection; the higher
the value, the more probable it is to go to the SWU-based OOV model for the
word class.

The following disambiguation symbols are printed out
- '#CLASS_NAME'
- if the probability-like value for OOV detection is > 0.0
-- '#CLASS_NAME_SWU'
-- '#CLASS_NAME_SWU_BACK'

The disambiguation symbols containing 'SWU' are neede to create an
SWU self-loop for OOV words.

Allowed options:
  --remove-wclass-prefix       : A string to be removed from word class labels (string,  default = "")

EOU

# command line options
my $wclass_list          = "";
my $remove_wclass_prefix = "";

# helper variables
my %word_list;

# get the optional command line options
GetOptions(
    "remove-wclass-prefix=s" => \$remove_wclass_prefix,
    ) or die "$Usage";

if (@ARGV != 1) {
    die $Usage;
}

# get the obligatory command line options
$wclass_list = shift @ARGV;

my %wc_disambig_swu_hash;
if (defined($wclass_list) and $wclass_list ne "") {
  open(WCL, "<$wclass_list") || die "$0: Error opening word class list file $wclass_list";
  # The input will be something like
  # C=CITYNAME<TAB>0.2
  # C=COUNTRY<TAB>0.0
  # ...
  while (<WCL>) {
    m/^\s*(.+)\s+(.+)\s*$/ || die "$0: bad line \"$_\"";
    my $word_class      = $1;
    my $wclass_oov_prob = $2;

    if ($wclass_oov_prob < 0.0 or $wclass_oov_prob > 1.0) {
      die "$0: The OOV probability $wclass_oov_prob for word class $word_class is not valid (needs to be a value in [0.0;1.0])";
    }

    my $filtered_word_class = $word_class;
    # now we remove the class prefix if it was set on the command line
    if (defined($remove_wclass_prefix) and $remove_wclass_prefix ne "") {
      # $remove_wclass_prefix is quoted so it is interpreted as text and not a regular expression
      $filtered_word_class =~ s/^\Q$remove_wclass_prefix\E//;
    }

    # printing out the disambiguation symbol for the word class
    print "#$filtered_word_class\n";

    # if the second entry in the line is a value x with 0.0 < x <= 1.0
    # there is an SWU model for OOV detection and we need more symbols
    if ($wclass_oov_prob > 0.0 and $wclass_oov_prob <= 1.0) {
      print "#${filtered_word_class}_SWU\n";
      print "#${filtered_word_class}_SWU_BACK\n";
    }
  }
  close(WCL);
}

