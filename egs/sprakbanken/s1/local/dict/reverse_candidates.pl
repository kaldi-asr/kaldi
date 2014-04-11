#!/usr/bin/perl

# This takes the output of e.g. get_candidate_prons.pl or limit_candidate_prons.pl,
# which is 7-tuples, one per line, of the form:

# word;pron;base-word;base-pron;rule-name;de-stress;rule-score
# (where rule-score is somtimes listed as optional, but this
# program does expect it, since we don't anticipate it being used
# without it).
# This program assumes that all the words and prons and rules have
# come from a reversed dictionary (reverse_dict.pl) where the order
# of the characters in the words, and the phones in the prons, have
# been reversed, and it un-reverses them.  That it, the characters
# in "word" and "base-word", and the phones in "pron" and "base-pron"
# are reversed; and the rule ("rule-name") is parsed as a 4-tuple,
# like:
# suffix,base-suffix,psuffix,base-psuffix
# so this program reverses the characters in "suffix" and "base-suffix"
# and the phones (separated by spaces) in "psuffix" and "base-psuffix".

sub reverse_str {
  $str = shift;
  return join("", reverse(split("", $str)));
}
sub reverse_pron {
  $str = shift;
  return join(" ", reverse(split(" ", $str)));
}

while(<>){ 
  chop;
  @A = split(";", $_);
  @A == 7 || die "Bad input line $_: found $len fields, expected 7.";

  ($word,$pron,$baseword,$basepron,$rule,$destress,$score) = @A;
  $word = reverse_str($word);
  $pron = reverse_pron($pron);
  $baseword = reverse_str($baseword);
  $basepron = reverse_pron($basepron);
  @R = split(",", $rule, 4);
  @R == 4 || die "Bad rule $rule";

  $R[0] = reverse_str($R[0]); # suffix.
  $R[1] = reverse_str($R[1]); # base-suffix.
  $R[2] = reverse_pron($R[2]); # pron.
  $R[3] = reverse_pron($R[3]); # base-pron.
  $rule = join(",", @R);
  @A = ($word,$pron,$baseword,$basepron,$rule,$destress,$score);
  print join(";", @A) . "\n";
}
