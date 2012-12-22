#!/usr/bin/perl
# A very small modification on ../../../tools/kaldi_lm/get_word_map.pl to account
# for no OOV vocab terms in timit.  - Navdeep Jaitly.


# This program reads in a file with one word
# on each line, and outputs a "translation file" of the form:
# word short-form-of-word
# on each line, 
# where short-form-of-word is a kind of abbreviation of the word.
#
# It uses the letters a-z and A-Z, plus the characters from 
# 128 to 255.  The first words in the file have the shortest representation.
#
# For convenience, it makes sure to give <s>, </s> 
# a consistent labeling, as A and B respectively.


# set up character table and some variables.
@C = ();
foreach $x (ord('A')...ord('Z')) { push @C, chr($x); }
foreach $x (ord('a')...ord('z')) {  push @C, chr($x); }
foreach $x(128...254) { push @C, chr($x); } # 255 is space so don't include it.

@index = ( 2 ); # array of indexes into @C... count up to [dim of C -1]
   # then add another index onto this.  Set it to 3, since 0 and 1 are
   # reserved for <s> and </s> respectively.

if (@ARGV != 2 && @ARGV != 3) {
  die "Usage: get_word_map.pl bos-symbol eos-symbol [words-in-order]\n";
}

$bos = shift @ARGV;
$eos = shift @ARGV;
print "$bos $C[0]\n";
print "$eos $C[1]\n";

sub get_short_form();

while(<>) {
  chop;
  $word = $_;
  $word =~ m:^\S+$: || die "Bad word $word";
  if($seen{$word}) { die "Word $word repeated"; }
  $seen{$word}++;
  if ($word ne $bos && $word ne $eos) {
    $short_form = get_short_form();
    print "$word $short_form\n";
  }
}

sub get_short_form() {
  $ans = "";
  foreach $i (@index) { $ans = $C[$i] . $ans; } # 
  # Now increment the index.
  $index[0]++;
  $cur_idx = 0;
  while ($index[$cur_idx] == @C) { # E.g. if the least significant digit
    # is out of the valid range... carry one.
    $index[$cur_idx] = 0;
    $cur_idx++;
    $index[$cur_idx]++; # This will extend the array if necessary.
  }
  return $ans;
}
