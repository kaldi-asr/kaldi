#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter

# makes unigram decoding-graph FSTs specific to each utterances, where the
# supplied top-n-words list together with the supervision text of the utterance are
# combined.

if (@ARGV != 1) {
  print STDERR "** Warning: this script is deprecated and will be removed.  See\n" .
               "** steps/cleanup/make_biased_lm_graphs.sh.\n" .
               "Usage: make_utterance_fsts.pl top-words-file.txt < text-archive > fsts-archive\n" .
               "e.g.: utils/sym2int.pl -f 2- data/lang/words.txt data/train/text | \\\n" .
               "  make_utterance_fsts.pl exp/foo/top_words.int | compile-train-graphs-fsts ... \n";
  exit(1);
}

($top_words_file) = @ARGV;

open(F, "<$top_words_file") || die "opening $top_words_file";

%top_word_probs = ( );

while(<F>) {
  @A = split;
  (@A == 2 && $A[0] > 0.0) || die "Bad line $_ in $top_words_file";
  $A[1] =~ m/^[0-9]+$/ || die "Expecting numeric word-ids in $top_words_file: $_\n";
  $top_word_probs{$A[1]} += $A[0];
}

while (<STDIN>) {
  @A = split;
  $utterance_id = shift @A;
  print "$utterance_id\n";
  $num_words = @A + 0;  # length of array @A
  %word_probs = %top_word_probs;
  foreach $w (@A) {
    $w =~ m/^[0-9]+$/ || die "Expecting numeric word-ids as stdin: $_";
    $word_probs{$w} += 1.0 / $num_words;
  }
  foreach $w (keys %word_probs) {
    $prob = $word_probs{$w};
    $prob > 0.0 || die "Word $w with bad probability $prob, utterance-id = $utterance_id\n";
    $cost = -log($prob);
    print "0 0 $w $w $cost\n";
  }
  $final_cost = -log(1.0 / $num_words);
  print "0 $final_cost\n";
  print "\n"; # Empty line terminates the FST in the text-archive format.
}
