#!/usr/bin/perl -w
# Copyright 2016 FAU Erlangen (Author: Axel Horndasch)

# Usage:  convert_to_embeddable_fst.pl [--back-transition <disambig-symbol>] < in.fst > out.fst
#
# This script modifies the input FST (in.fst) so that it is not possible to go
# through the graph producing only empty words (<eps>).
# If the FST is a unigram with self-loops the script converts it to a unigram
# with separate states. As a result, it is not possible to go through the graph
# producing only empty words (<eps>).
#
# If the option --back-transition <disambig-symbol> is used and the original
# FST was a unigram with self-loops, a back transition is added to the FST
# which allows to go through the graph one or more times. This is useful if the
# terminal symbols are e.g. sub-word units (SWUs) which are used to
# detect/cover out-of-vocabulary words (OOVs).

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

# For n-grams where n > 1 (bigrams, trigrams, 4-grams etc.) the strategy is the
# following:
# - Find all transitions from the start state to other states (here defined as
#   "second_states") by looking for sth. like "<s>     <s>"
# - Find all transitions from "second states" to the end state and remove them.

use strict;
use warnings;
use Getopt::Long;

my $back_trans_disambig_symbol = "";

# get the command line option values
GetOptions (
    "back-transition=s"   => \$back_trans_disambig_symbol,
    );

# helper variables
my $has_initial_transition = 0;
my $is_self_loop_unigram   = 1;
my $has_final_transition   = 0;
my $has_final_state        = 0;

my $state_nr_one           = "";
my $state_nr_two           = "";

my $transition             = "";

my $first_state            = "";
my @initial_transitions    = ();
my %second_states          = ();
my @internal_transitions   = ();
my @final_transitions      = ();
my %final_states           = ();

# Go through the FST transition by transition (== line by line)
while ($transition = <STDIN>) {
  chomp($transition);

  # This transition is the beginning of a sentence (BOS, <s>)
  if ($transition =~ /^(\d+)\t(\d+)\t<s>\t<s>(.*)$/) {
    $state_nr_one = $1;
    $state_nr_two = $2;
    push(@initial_transitions, $transition);
    $second_states{$state_nr_two} = 1;
    if ($first_state eq "") {
      $first_state = $state_nr_one;
    } elsif($first_state ne $state_nr_one) {
      warn "Found initial transition, but its first state \"$state_nr_one\" is not equal to the first state \"$first_state\" found before";
    }
    $has_initial_transition = 1;
  # This transition is the end of a sentence (EOS, </s>)
  } elsif ($transition =~ /^(\d+)\t(\d+)\t<\/s>\t<\/s>(.*)$/) {
    $state_nr_one = $1;
    $state_nr_two = $2;
    push(@final_transitions, $transition);
    $final_states{$state_nr_two} = 1;
    $has_final_transition = 1;
  # The following transitions are either a unigram self-loop or transitions of a higher order n-gram
  # (because they contain no '<s>' or '</s>')
  } elsif ($transition =~ /(\d+)\t(\d+)\t(\S+)\t(\S+)\t(\d+\.\d+)$/) {
    $state_nr_one = $1;
    $state_nr_two = $2;
    push(@internal_transitions, $transition);
    if($state_nr_one != $state_nr_two) {
      $is_self_loop_unigram = 0;
    }
  # The final state should be a single number
  } elsif ($transition =~ /^(\d+)(\s*)$/) {
    $final_states{$1} = 1;
    $has_final_state = 1;
  }
}

if (not ($has_initial_transition and $has_final_transition and $has_final_state)) {
  warn "$0: The graph to be converted did not meet the requirements, please check ...";
}

# At this point we know if this is a self-loop unigram or an n-gram with n > 2,
# as a consequence we can go through all transitions modify them (if necessary)
# and print them out (unless we want to delete them).

foreach $transition (@initial_transitions) {
  if(not $is_self_loop_unigram) {
    print "$transition\n";
  } elsif ($transition =~ /^(\d+)\t(\d+)\t<s>\t<s>(.*)$/) {
    print "0\t1\t<s>\t<s>$3\n";
  } else {
    die "$0: Something went wrong with this initial transition: $transition\n";
  }
}

foreach $transition (@internal_transitions) {
  if(not $is_self_loop_unigram) {
    print "$transition\n";
  } elsif ($transition =~ /(\d+)\t(\d+)\t(\S+)\t(\S+)\t(\d+\.\d+)$/) {
    print "1\t2\t$3\t$4\t$5\n";
  } else {
    die "$0: Something went wrong with this internal transition: $transition\n";
  }
}

# Add the back transition if a disambiguation symbol was provided
# on the command line and if this is a unigram self-loop
if ($is_self_loop_unigram and $back_trans_disambig_symbol ne "") {
  print "2\t1\t$back_trans_disambig_symbol\t<eps>\n";
}

foreach $transition (@final_transitions) {
  if ($transition =~ /^(\d+)\t(\d+)\t<\/s>\t<\/s>(.*)$/) {
    if($is_self_loop_unigram) {
      print "2\t3\t</s>\t</s>$3\n";
    } else {
      if(not defined($second_states{$1})) {
	print "$transition\n";
      }
    }
  } else {
    die "$0: Something went wrong with this final transition: $transition\n";
  }
}

if($is_self_loop_unigram) {
  print "3\n";
} else {
  foreach my $state (keys %final_states) {
    print "$state\n";
  }
}

