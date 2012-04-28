#!/usr/bin/perl


if ($ARGV[0] eq "-f") {
  shift @ARGV; 
  $field_spec = shift @ARGV; 
  if ($field_spec =~ m/^\d+$/) {
    $field_begin = $field_spec - 1; $field_end = $field_spec - 1;
  }
  if ($field_spec =~ m/^(\d*)[-:](\d*)/) { # accept e.g. 1:10 as a courtesty (properly, 1-10)
    if ($1 ne "") {
      $field_begin = $1 - 1;    # Change to zero-based indexing.
    }
    if ($2 ne "") {
      $field_end = $2 - 1;      # Change to zero-based indexing.
    }
  }
  if (!defined $field_begin && !defined $field_end) {
    die "Bad argument to -f option: $field_spec"; 
  }
}


while (<>) {
  @A = split(" ", $_);
  for ($n = 0; $n < @A; $n++) {
    $a = $A[$n];
    if ( (!defined $field_begin || $n >= $field_begin)
         && (!defined $field_end || $n <= $field_end)) {
      $a =~ s:(|\-)^\[LAUGHTER-(.+)\](|\-)$:$1$2$3:; # e.g. [LAUGHTER-STORY] -> STORY;
      # $1 and $3 relate to preserving trailing "-"
      $a =~ s:^\[(.+)/.+\](|\-)$:$1$2:; # e.g. [IT'N/ISN'T] -> IT'N ... note,
      # 1st part may include partial-word stuff, which we process further below,
      # e.g. [LEM[GUINI]-/LINGUINI]
      # the (|\_) at the end is to accept and preserve trailing -'s.
      $a =~ s:^(|\-)\[[^][]+\](.+)$:-$2:; # e.g. -[AN]Y , note \047 is quote;
      # let the leading - be optional on input, as sometimes omitted.
      $a =~ s:^(.+)\[[^][]+\](|\-)$:$1-:; # e.g. AB[SOLUTE]- -> AB-;
      # let the trailing - be optional on input, as sometimes omitted.
      $a =~ s:([^][]+)\[.+\]$:$1:; # e.g. EX[SPECIALLY]-/ESPECIALLY] -> EX-
      # which is a  mistake in the input.
      $a =~ s:^\{(.+)\}$:$1:;                # e.g. {YUPPIEDOM} -> YUPPIEDOM
      $a =~ s:[A-Z]\[([^][])+\][A-Z]:$1-$3:; # e.g. AMMU[N]IT- -> AMMU-IT-
      $a =~ s:_\d$::;                        # e.g. THEM_1 -> THEM 
    }
    $A[$n] = $a;
  }
  print join(" ", @A) . "\n";
}
