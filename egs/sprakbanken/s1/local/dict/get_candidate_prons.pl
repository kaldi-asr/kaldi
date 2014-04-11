#!/usr/bin/perl

# This script takes three command-line arguments (typically files, or "-"):
# the suffix rules (as output by get_rules.pl), the rule-hierarchy 
# (from get_rule_hierarchy.pl), and the words that we want prons to be 
# generated for (one per line).

# The output consists of candidate generated pronunciations for those words,
# together with information about how we generated those pronunciations.
# This does not do pruning of the candidates using the restriction
# "you can't use a more general rule when a more specific one is applicable".
# That is done by limit_candidate_prons.pl.

# Each line of the output consists of a 4-tuple, separated by ";", of the
# form:
# word;pron;base-word;base-pron;rule-name;destress[;rule-score]
# [the last field is only present if you supplied rules with score information].
# where:
# - "word" is the input word that we queried for, e.g. WASTED
# - "pron" is the generated pronunciation, e.g. "W EY1 S T AH0 D"
# - rule-name is a 4-tuple separated by commas that describes the rule, e.g.
#   "STED,STING,D,NG",
# - "base-word" is the base-word we're getting the pron from,
#   e.g. WASTING
# - "base-pron" is the pron of the base-word, e.g. "W EY1 S T IH0 NG"
# - "destress" is either "yes" or "no" and corresponds to whether we destressed the
#   base-word or not [de-stressing just corresponds to just taking any 2's down to 1's,
#   although we may extend this in future]... 
# - "rule-score" is a numeric score of the rule (this field is only present
#   if there was score information in your rules.


(@ARGV == 2  || @ARGV == 3) || die "Usage: get_candidate_prons.pl rules base-dict [ words ]";

$min_prefix_len = 3;  # this should probably match with get_rules.pl

$rules = shift @ARGV; # Note: rules may be with destress "yes/no" indicators or without...
                      # if without, it's treated as if both "yes" and "no" are present.
$dict = shift @ARGV;

open(R, "<$rules") || die "Opening rules file: $rules";

sub process_word;

while(<R>) {
  chop $_;
  my ($rule, $destress, $rule_score) = split(";", $_); # We may have "destress" markings (yes|no),
  # and scores, or we may have just rule, in which case
  # $destress and $rule_score will be undefined.

  my @R = split(",", $rule, 4); # "my" means new instance of @R each
  # time we do this loop -> important because we'll be creating
  # a reference to @R below.
  # Note: the last arg to SPLIT tells it how many fields max to get.
  # This stops it from omitting empty trailing fields.
  @R == 4 || die "Bad rule $_";
  $suffix = $R[0]; # Suffix of word we want pron for.
  if (!defined $isrule{$rule}) {
    $isrule{$rule} = 1; # make sure we do this only once for each rule 
    # (don't repeate for different stresses).
    if (!defined $suffix2rule{$suffix}) {
      # The syntax [ $x, $y, ... ] means a reference to a newly created array
      # containing $x, $y, etc.   \@R creates an array reference to R.
      # so suffix2rule is a hash from suffix to ref to array of refs to 
      # 4-dimensional arrays.
      $suffix2rule{$suffix} = [ \@R ];
    } else {
      # Below, the syntax @{$suffix2rule{$suffix}} dereferences the array
      # reference inside the hash; \@R pushes onto that array a new array
      # reference pointing to @R.
      push @{$suffix2rule{$suffix}}, \@R;
    }
  }
  if (!defined $rule_score) { $rule_score = -1; } # -1 means we don't have the score info.
  
  # Now store information on which destress markings (yes|no) this rule
  # is valid for, and the associated scores (if supplied)
  # If just the rule is given (i.e. no destress marking specified),
  # assume valid for both.
  if (!defined $destress) { # treat as if both "yes" and "no" are valid.
    $rule_and_destress_to_rule_score{$rule.";yes"} = $rule_score;
    $rule_and_destress_to_rule_score{$rule.";no"} = $rule_score;
  } else {
    $rule_and_destress_to_rule_score{$rule.";".$destress} = $rule_score;
  }

}

open(D, "<$dict") || die "Opening base dictionary: $dict";
while(<D>) {
  @A = split(" ", $_);
  $word = shift @A;
  $pron = join(" ", @A);
  if (!defined $word2prons{$word}) {
    $word2prons{$word} = [ $pron ]; # Ref to new anonymous array containing just "pron".
  } else {
    push @{$word2prons{$word}}, $pron; # Push $pron onto array referred to (@$ref derefs array).
  }
}
foreach $word (%word2prons) {
  # Set up the hash "prefixcount", which says how many times a char-sequence
  # is a prefix (not necessarily a strict prefix) of a word in the dict.
  $len = length($word);
  for ($l = 0; $l <= $len; $l++) {
    $prefixcount{substr($word, 0, $l)}++;
  }
}

open(R, "<$rules") || die "Opening rules file: $rules";


while(<>) {
  chop;
  m/^\S+$/ || die;
  process_word($_);
}

sub process_word {
  my $word = shift @_;
  $len = length($word);
  # $owncount is used in evaluating whether a particular prefix is a prefix
  # of some other word in the dict... if a word itself may be in the dict
  # (usually because we're running this on the dict itself), we need to
  # correct for this.
  if (defined $word2prons{$word}) { $owncount = 1; } else { $owncount = 0; }
  
  for ($prefix_len = $min_prefix_len; $prefix_len <= $len; $prefix_len++) {
    my $prefix = substr($word, 0, $prefix_len);
    my $suffix = substr($word, $prefix_len);
    if ($prefixcount{$prefix} - $owncount == 0) {
      # This prefix is not a prefix of any word in the dict, so no point
      # checking the rules below-- none of them can match.
      next;
    }
    $rules_array_ref = $suffix2rule{$suffix};
    if (defined $rules_array_ref) {
      foreach $R (@$rules_array_ref) { # @$rules_array_ref dereferences the array.
        # $R is a refernce to a 4-dimensional array, whose elements we access with
        # $$R[0], etc.
        my $base_suffix = $$R[1];
        my $base_word = $prefix . $base_suffix;
        my $base_prons_ref = $word2prons{$base_word};
        if (defined $base_prons_ref) {
          my $psuffix = $$R[2];
          my $base_psuffix = $$R[3];
          if ($base_psuffix ne "") { 
            $base_psuffix = " " . $base_psuffix; 
            # Include " ", the space between phones, to prevent
            # matching partial phones below.
          }
          my $base_psuffix_len = length($base_psuffix);
          foreach $base_pron (@$base_prons_ref) { # @$base_prons_ref derefs 
            # that reference to an array.
            my $base_pron_prefix_len = length($base_pron) - $base_psuffix_len;
            # Note: these lengths are in characters, not phones.
            if ($base_pron_prefix_len >= 0 && 
                substr($base_pron, $base_pron_prefix_len) eq $base_psuffix) {
              # The suffix of the base_pron is what it should be.
              my $pron_prefix = substr($base_pron, 0, $base_pron_prefix_len);
              my $rule = join(",", @$R); # we'll output this..
              my $len = @R;
              for ($destress = 0; $destress <= 1; $destress++) { # Two versions 
                # of each rule: with destressing and without.
                # pron is the generated pron.
                if ($destress) {  $pron_prefix =~ s/2/1/g; }
                my $pron;
                if ($psuffix ne "") { $pron = $pron_prefix . " " . $psuffix; }
                else { $pron = $pron_prefix; }
                # Now print out the info about the generated pron.
                my $destress_mark = ($destress ? "yes" : "no");
                my $rule_score = $rule_and_destress_to_rule_score{$rule.";".$destress_mark};
                if (defined $rule_score) { # Means that the (rule,destress) combination was
                  # seen [note: this if-statement may be pointless, as currently we don't
                  # do any pruning of rules].
                  my @output = ($word, $pron, $base_word, $base_pron, $rule, $destress_mark);
                  if ($rule_score != -1) { push @output, $rule_score; } # If scores were supplied,
                  # we also output the score info.
                  print join(";", @output) . "\n";
                }
              }
            }  
          }
        }
      }
    }
  }
}  
