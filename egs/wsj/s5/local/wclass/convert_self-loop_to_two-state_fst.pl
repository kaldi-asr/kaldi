#!/usr/bin/perl -w
# Copyright 2016 FAU Erlangen (Author: Axel Horndasch)

# Usage:  convert_self-loop_to_two-state_fst.pl [--back-transition <disambig-symbol>] < in.fst > out.fst
#
# This script converts a unigram with self-loops to a unigram with separate
# states. As a result, it is not possible to go through the graph producing
# only empty words (<eps>).
#
# If the option --back-transition <disambig-symbol> is used, a back transition
# is added to the FST which allows to go through the graph one ore more times.
# This is useful if the terminal symbols are e.g. sub-word units (SWUs) which
# are used to detect/cover out-of-vocabulary words (OOVs).

### 0       2       <s>     <s>     0.693359375
### 1
### 2       2       Kate    Kate    1.79199219
### 2       2       Cay     Cay     1.09863281
### 2       1       </s>    </s>

# should become

### 0       1       <s>     <s>     0.693359375
### 1       2       Cay     Cay     1.09863281
### 1       2       Kate    Kate    1.79199219
### 2       3       </s>    </s>
### 3
### optional (with --back-transition #FIRSTNAME_BACK)
### 2       1       #FIRSTNAME_BACK <eps>

# -> no more self-loops

use strict;
use warnings;
use Getopt::Long;

my $back_trans_disambig_symbol = "";

my %optctl = (
    "back-transition"   => \$back_trans_disambig_symbol,
    );

# get the option values
&GetOptions(\%optctl,
    "back-transition=s",
    );

# states of the current transition in the FST
my $state_nr_one;
my $state_nr_two;

# helper variables
my $has_initial_transition = 0;
my $has_self_loops         = 0;
my $has_final_transition   = 0;
my $has_final_state        = 0;

# Go through the FST transition by transition (== line by line)
while (<STDIN>){
  # This transition is the beginning of a sentence (BOS, <s>)
  if (/^(\d+)\t(\d+)\t<s>\t<s>(.*)$/) {
    $state_nr_one = $1;
    $state_nr_two = $2;
    print "0\t1\t<s>\t<s>$3\n";
    if ($state_nr_one == $state_nr_two) {
      warn "Found initial transition, but it is a self-loop and it shouldn't be, please check: $state_nr_one";
    }
    $has_initial_transition = 1;
  # This transition is the end of a sentence (EOS, </s>)
  } elsif (/^(\d+)\t(\d+)\t<\/s>\t<\/s>(.*)$/) {
    $state_nr_one = $1;
    $state_nr_two = $2;
    print "2\t3\t</s>\t</s>$3\n";
    if ($state_nr_one == $state_nr_two) {
      warn "Found final transition but it is a self-loop and it shouldn't be, please check: $state_nr_one";
    }
    $has_final_transition = 1;
  # This transition is assumed to be the self-loop
  # (because it contains no '<s>' or '</s>')
  } elsif (/(\d+)\t(\d+)\t(\S+)\t(\S+)\t(\d+\.\d+)$/) {
    if ($1 == $2) {
      print "1\t2\t$3\t$4\t$5\n";
    } else {
      warn "This transition should be a self-loop, but it is not: $2 -> $4\n";
    }
    $has_self_loops = 1;
  # The final state is usually a single number
  } elsif (/^(\d+)(\s*)$/) {
    print "3\n";
    $has_final_state = 1;
  }
}

if (not ($has_initial_transition and $has_self_loops and $has_final_transition and $has_final_state)) {
  warn "$0: The graph to be converted did not meet the requirements, please check ...";
}

# Add the back transition if a disambiguation symbol was provided
# on the command line
if ($back_trans_disambig_symbol ne "") {
  print "2\t1\t$back_trans_disambig_symbol\t<eps>\n";
}
