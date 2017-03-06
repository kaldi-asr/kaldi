#!/usr/bin/perl -w
# Copyright 2016 FAU Erlangen (Author: Axel Horndasch)
# Apache 2.0.
#
use strict;
use warnings;
use Getopt::Long;

my $Usage = <<EOU;
Usage:  $0 [options] <wclass-name> <wclass-label> <wclass-list> <all-wclass-count> <one-wclass-count>

This script takes a file with the replacement counts of entries for all word
classes and extracts the counts for a specific word class.

In case the probability-like value of the word class for OOV detection is more
than 0.0 (read from <wclass-list>), this script adds a count for a special
sub-word unit (SWU) symbol. That special SWU can later be replaced by an SWU
graph for OOV detection.

Allowed options:
  --replace-blanks-with-dash   : Replace blanks in word class entry with "-"   (boolean, default = false)

EOU

# command line options
my $wclass_name              = "";
my $wclass_label             = "";
my $wclass_list              = "";
my $all_wclass_count_file    = "";
my $one_wclass_count_file    = "";

my $replace_blanks_with_dash = "false";

# get the optional command line options
GetOptions(
    'replace-blanks-with-dash=s' => \$replace_blanks_with_dash,
    ) or die "$Usage";

($replace_blanks_with_dash eq "true" || $replace_blanks_with_dash eq "false") ||
  die "$0: Bad value for option --replace-blanks-with-dash\n";

if (@ARGV != 5) {
    die $Usage;
}

# get the obligatory command line options
$wclass_name           = shift @ARGV;
$wclass_label          = shift @ARGV;
$wclass_list           = shift @ARGV;
$all_wclass_count_file = shift @ARGV;
$one_wclass_count_file = shift @ARGV;

my $wclass_oov_prob;
undef($wclass_oov_prob);

# opening the word class list
if (defined($wclass_list) and $wclass_list ne "") {
  open(WCL, "<$wclass_list") || die "Error opening word class list file $wclass_list";
  # The content of file 'wclass_list.txt' should look similar to this:
  # C=CITYNAME<TAB>0.2
  # C=COUNTRY<TAB>0.0
  # ...
  while (<WCL>) {
    m/^\s*(.+)\s+(.+)\s*$/ || die "$0: bad line \"$_\"";
    my $label_from_wclass_list = $1;
    $wclass_oov_prob           = $2;

    # break out of the loop if we found the word class in the list
    if ($label_from_wclass_list eq $wclass_label) {
      last;
    }
  }
  close(WCL);
} else {
  die "$0: Please specify a valid 'wclass_list.txt' on the command line, exiting...\n$Usage";
}

# if $wclass_oov_prob is still undefined, the class we were looking for could not be found
if (not defined($wclass_oov_prob)) {
  die "$0: Couldn't find word class label \"$wclass_label\" in word class list \"$wclass_list\", exiting...";
}

# if the OOV probability is not in the valid range -> exit with an error
if (not ($wclass_oov_prob >= 0.0 and $wclass_oov_prob <= 1.0)) {
  die "$0: The probability for OOV words \"$wclass_oov_prob\" for word class " .
    "\"$wclass_name\" is not valid (needs to be a value in [0.0;1.0]), exiting...";
}

# At this point we have the word class label and a valid OOV probability. We
# now need to create a class-specific counts file. And if the OOV probability
# is > 0.0 we also add the entry ${wclass_label}_SWU; the number (or weight)
# attached to the ${wclass_label}_SWU entry depends on the overall count of
# entries in the specific word class and the OOV probability read from
# $wclass_list.

open(OWCF, ">$one_wclass_count_file") ||
  die "Error opening count file \"$one_wclass_count_file\" for word class \"$wclass_name\" for writing, exiting...";

# Check if OOV detection probability is set to 1.0. If so, only SWUs are
# allowed in the word class. No other entries apart from the special sub-word
# unit (SWU) symbol (${wclass_label}_SWU, e.g. C=WEEKDAY_SWU), must occur in
# the word class-based sub-language model.
# This basically means that it is assumed that there will only be unknown
# words for this word class.
if ($wclass_oov_prob == 1.0) {
  print OWCF "1 ${wclass_label}_SWU\n";
} else {
  # opening the counts file which contains all classes
  if (defined($all_wclass_count_file) and $all_wclass_count_file ne "") {
    open(AWCF, "<$all_wclass_count_file") ||
      die "Error opening word class count file \"$all_wclass_count_file\" for reading, exiting...";

    my $overall_class_entry_count = 0;
    # The content of $all_wclass_count_file should look similar to this:
    # ...
    # C=WEEKDAY 3487 THURSDAY
    # C=WEEKDAY 5662 TUESDAY
    # C=WEEKDAY 4794 WEDNESDAY
    # C=US_STATE 885 ALABAMA
    # C=US_STATE 963 ALASKA
    # C=US_STATE 1835 ARIZONA
    # ...
    while (<AWCF>) {
      m/^\s*(\S*)\s+(\d+)\s+(.+)\s*$/ || die "$0: Bad line \"$_\" in count class file \"$all_wclass_count_file\", exiting...";
      my $label_from_counts_file  = $1;
      my $count_for_class_entry   = $2;
      my $class_entry             = $3;
      # We only want the entries with the word class label with which
      # this script was called ($wclass_label)
      if ($label_from_counts_file eq $wclass_label) {
        # We preserve the old counts and we are interested in the overall count
        # to be able to compute the right ${wclass_label}_SWU (== OOV) count, if
        # applicable (see below).
        if($replace_blanks_with_dash eq "true") {
          $class_entry =~ s/ /-/g;
        }
        print OWCF "$count_for_class_entry $class_entry\n";
        $overall_class_entry_count += $count_for_class_entry;
      }
    }
    close(AWCF);

    # Check if OOV detection using sub-word units should be added.
    if ($wclass_oov_prob > 0.0) {
      my $SWU_OOV_count = int($overall_class_entry_count * $wclass_oov_prob / (1.0-$wclass_oov_prob));
      print OWCF "$SWU_OOV_count ${wclass_label}_SWU\n";
    }
  } else {
    die "$0: Something went wrong with the all counts file \"$all_wclass_count_file\" exiting...";
  }
}
close(OWCF);

