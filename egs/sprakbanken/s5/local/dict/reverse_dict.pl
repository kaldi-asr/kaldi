#!/usr/bin/perl

# Used in conjunction with get_rules.pl
# example input line: XANTHE  Z AE1 N DH
# example output line: EHTNAX DH N AE1 Z

while(<>){ 
  @A = split(" ", $_);
  $word = shift @A;
  $word = join("", reverse(split("", $word))); # Reverse letters of word.
  @A = reverse(@A); # Reverse phones in pron.
  unshift @A, $word;
  print join(" ", @A) . "\n";
}
