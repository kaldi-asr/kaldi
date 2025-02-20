#!/usr/bin/env perl

#This reads in rules, of the form put out by get_rules.pl, e.g.:
# ERT,,ER0 T,
# MENT,ING,M AH0 N T,IH0 NG
# S,TON,Z,T AH0 N
# ,ER,IH0 NG,IH0 NG ER0
# ,'S,M AH0 N,M AH0 N Z
#TIONS,TIVE,SH AH0 N Z,T IH0 V

# and it works out a hierarchy that says which rules are sub-cases
# of which rules: it outputs on each line a pair separated by ";", where
# each member of the pair is a rule, first one is the specialization, the
# second one being more general.
# E.g.:
# RED,RE,D,/ED,E,D,
# RED,RE,D,/D,,D,
# GING,GE,IH0 NG,/ING,I,IH0 NG,
# TOR,TING,T ER0,T IH0 NG/OR,OR,T ER0,T ER0 
# ERED,ER,D,/RED,R,D,
# ERED,ER,D,/ED,,D,




while(<>) {
  chop;
  $rule = $_;
  $isrule{$rule} = 1;
  push @rules, $rule;
}

foreach my $rule (@rules) {
  # Truncate the letters and phones in the rule, while we
  # can, to get more general rules; if the more general rule
  # exists, put out the pair.
  @A = split(",", $rule);
  @suffixa = split("", $A[0]);
  @suffixb = split("", $A[1]);
  @psuffixa = split(" ", $A[2]);
  @psuffixb = split(" ", $A[3]);
  for ($common_suffix_len = 0; $common_suffix_len < @suffixa && $common_suffix_len < @suffixb;) {
    if ($suffixa[$common_suffix_len] eq $suffixb[$common_suffix_len]) {
      $common_suffix_len++;
    } else {
      last;
    }
  }
  for ($common_psuffix_len = 0; $common_psuffix_len < @psuffixa && $common_psuffix_len < @psuffixb;) {
    if ($psuffixa[$common_psuffix_len] eq $psuffixb[$common_psuffix_len]) {
      $common_psuffix_len++;
    } else {
      last;
    }
  }
  # Get all combinations of pairs of integers <= (common_suffix_len, common_psuffix_len),
  # except (0,0), and print out this rule together with the corresponding rule (if it exists).
  for ($m = 0; $m <= $common_suffix_len; $m++) {
    $sa = join("", @suffixa[$m...$#suffixa]); # @x[a..b] is array slice notation.
    $sb = join("", @suffixb[$m...$#suffixb]);
    for ($n = 0; $n <= $common_psuffix_len; $n++) {
      if (!($m == 0 && $n == 0)) {
        $psa = join(" ", @psuffixa[$n...$#psuffixa]);
        $psb = join(" ", @psuffixb[$n...$#psuffixb]);
        $more_general_rule = join(",", ($sa, $sb, $psa, $psb));
        if (defined $isrule{$more_general_rule}) {
          print $rule . ";" . $more_general_rule . "\n";
        }
      }
    }
  }
}

