#!/usr/bin/perl


while(<>) {
  $w = $_; chop $w; $w_orig = $w;
  $w =~ s:(|\-)^\[LAUGHTER-(.+)\](|\-)$:$1$2$3:;  # e.g. [LAUGHTER-STORY] -> STORY;
  # $1 and $3 relate to preserving trailing "-"
  $w =~ s:^\[(.+)/.+\](|\-)$:$1$2:; # e.g. [IT'N/ISN'T] -> IT'N ... note,
  # 1st part may include partial-word stuff, which we process further below,
  # e.g. [LEM[GUINI]-/LINGUINI]
  # the (|\_) at the end is to accept and preserve trailing -'s.
  $w =~ s:^(|\-)\[[^][]+\](.+)$:-$2:;  # e.g. -[AN]Y , note \047 is quote;
  # let the leading - be optional on input, as sometimes omitted.
  $w =~ s:^(.+)\[[^][]+\](|\-)$:$1-:;  # e.g. AB[SOLUTE]- -> AB-;
  # let the trailing - be optional on input, as sometimes omitted.
  $w =~ s:([^][]+)\[.+\]$:$1:; # e.g. EX[SPECIALLY]-/ESPECIALLY] -> EX-
  # which is a  mistake in the input.
  $w =~ s:^\{(.+)\}$:$1:; # e.g. {YUPPIEDOM} -> YUPPIEDOM
  $w =~ s:[A-Z]\[([^][])+\][A-Z]:$1-$3:; # e.g. AMMU[N]IT- -> AMMU-IT-
  $w =~ s:_\d$::;  # e.g. THEM_1 -> THEM
  print "$w_orig $w\n";
}
