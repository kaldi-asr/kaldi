#!/usr/bin/perl

# This program creates suggested suffix rules from a dictionary.
# It outputs quadruples of the form:
# suffix,base-suffix,psuffix,base-psuffix
# where "suffix" is the suffix of the letters of a word, "base-suffix" is
# the suffix of the letters of the base-word, "psuffix" is the suffix of the
# pronunciation of the word (a space-separated list of phonemes), and
# "base-psuffix" is the suffix of the pronunciation of the baseword.
# As far as this program is concerned, there is no distinction between
# "word" and "base-word".  To simplify things slightly, what it does
# is return all tuples (a,b,c,d) [with a != b] such that there are
# at least $min_suffix_count instances in the dictionary of
# a (word-prefix, pron-prefix) pair where there exists (word,pron)
# pairs of the form
# ( word-prefix . a,  pron-prefix . c)
# and 
# ( word-prefix . b, pron-prefix . d)
# For example if (a,b,c,d) equals (USLY,US,S L IY0,S)
# then this quadruple will be output as long as there at least
# e.g. 30 instances of prefixes like (FAM, F EY1 M AH0)
# where there exist (word, pron) pairs like:
# FAMOUS, F EY1 M AH0 S
# FAMOUSLY  F EY1 M AH0 S L IY0
#
# There are some modifications to the picture above, for efficiency.
# If $disallow_empty_suffix != 0, this program will not output 4-tuples where
# the first element (the own-word suffix) is empty, as this would cause
# efficiency problems in get_candidate_prons.pl.  If 
# $ignore_prefix_stress != 0, this program will ignore stress markings
# while evaluating whether prefixes are the same.
# The minimum count for a quadruple to be output is $min_suffix_count
# (e.g. 30).
#
# The function of this program is not to evaluate the accuracy of these rules;
# it is mostly a pruning step, where we suggest rules that have large enough
# counts to be suitable for our later procedure where we evaluate their
# accuracy in predicting prons.

$disallow_empty_suffix = 1; # Disallow rules where the suffix of the "own-word" is
   # empty.  This is for efficiency in later stages (e.g. get_candidate_prons.pl).
$min_prefix_len = 3;  # this must match with get_candidate_prons.pl
$ignore_prefix_stress = 1; # or 0 to take account of stress in prefix.
$min_suffix_count = 20;

# Takes in dictionary.

print STDERR "Reading dict\n";
while(<>) {
  @A = split(" ", $_);
  my $word = shift @A;
  my $pron = join(" ", @A);
  if (!defined $prons{$word}) {
    $prons{$word} = $pron;
    push @words, $word;
  } else {
    $prons{$word} = $prons{$word} . ";" . $pron;
  }
}

# Get common suffixes (e.g., count >100).  Include empty suffix.

print STDERR "Getting common suffix counts.\n";
{
  foreach $word (@words) {
    $len = length($word);
    for ($x = $min_prefix_len; $x <= $len; $x++) {
      $suffix_count{substr($word, $x)}++;
    }
  }

  foreach $suffix (keys %suffix_count) {
    if ($suffix_count{$suffix} >= $min_suffix_count) {
      $newsuffix_count{$suffix} = $suffix_count{$suffix};
    }
  }
  %suffix_count = %newsuffix_count;
  undef %newsuffix_count;

  foreach $suffix ( sort { $suffix_count{$b} <=> $suffix_count{$a} } keys %suffix_count ) {
    print STDERR "$suffix_count{$suffix} $suffix\n";
  }
}

print STDERR "Getting common suffix pairs.\n";

{
  print STDERR " Getting map from prefix -> suffix-set.\n";

  # Create map from prefix -> suffix-set.
  foreach $word (@words) {
    $len = length($word);
    for ($x = $min_prefix_len; $x <= $len; $x++) {
      $prefix = substr($word, 0, $x);
      $suffix = substr($word, $x);
      if (defined $suffix_count{$suffix}) { # Suffix is common...
        if (!defined $suffixes_of{$prefix}) {
          $suffixes_of{$prefix} = [ $suffix ]; # Create a reference to a new array with
          # one element.
        } else {
          push @{$suffixes_of{$prefix}}, $suffix; # Push $suffix onto array that the
          # hash member is a reference .
        }
      }
    }
  }
  my %suffix_set_count;
  print STDERR " Getting map from suffix-set -> count.\n";
  while ( my ($key, $value) = each(%suffixes_of) ) { 
    my @suffixes = sort ( @$value );
    $suffix_set_count{join(";", @suffixes)}++;
  }
  print STDERR " Getting counts for suffix pairs.\n";
  while ( my ($suffix_set, $count) = each (%suffix_set_count) ) {
    my @suffixes = split(";", $suffix_set);
    # Consider pairs to be ordered.  This is more convenient
    # later on.
    foreach $suffix_a (@suffixes) {
      foreach $suffix_b (@suffixes) {
        if ($suffix_a ne $suffix_b) {
          $suffix_pair = $suffix_a . "," . $suffix_b;
          $suffix_pair_count{$suffix_pair} += $count;
        }
      }
    }
  }

  # To save memory, only keep pairs above threshold in the hash.
  while ( my ($suffix_pair, $count) = each (%suffix_pair_count) ) {
    if ($count >= $min_suffix_count) {
      $new_hash{$suffix_pair} = $count;
    }
  }
  %suffix_pair_count = %new_hash;
  undef %new_hash;

  # Print out the suffix pairs so the user can see.
  foreach $suffix_pair ( 
      sort { $suffix_pair_count{$b} <=> $suffix_pair_count{$a} } keys %suffix_pair_count ) {
    print STDERR "$suffix_pair_count{$suffix_pair} $suffix_pair\n";
  }
}

print STDERR "Getting common suffix/suffix/psuffix/psuffix quadruples\n";

{
  while ( my ($prefix, $suffixes_ref) = each(%suffixes_of) ) {
    # Note: suffixes_ref is a reference to an array.  We dereference with
    # @$suffixes_ref.
    # Consider each pair of suffixes (in each order).
    foreach my $suffix_a ( @$suffixes_ref ) {
      foreach my $suffix_b ( @$suffixes_ref ) {
        # could just used "defined" in next line, but this is for clarity.
        $suffix_pair = $suffix_a.",".$suffix_b;
        if ( $suffix_pair_count{$suffix_pair} >= $min_suffix_count ) {
          foreach $pron_a_str (split(";", $prons{$prefix.$suffix_a})) {
            @pron_a = split(" ", $pron_a_str);
            foreach $pron_b_str (split(";", $prons{$prefix.$suffix_b})) {
              @pron_b = split(" ", $pron_b_str);
              $len_a = @pron_a; # evaluating array as scalar automatically gives length.
              $len_b = @pron_b;
              for (my $pos = 0; $pos <= $len_a && $pos <= $len_b; $pos++) {
                # $pos is starting-pos of psuffix-pair. 
                $psuffix_a = join(" ", @pron_a[$pos...$#pron_a]);
                $psuffix_b = join(" ", @pron_b[$pos...$#pron_b]);
                $quadruple = $suffix_pair . "," . $psuffix_a . "," . $psuffix_b;
                $quadruple_count{$quadruple}++;
                
                my $pron_a_pos = $pron_a[$pos], $pron_b_pos = $pron_b[$pos];
                if ($ignore_prefix_stress) {
                  $pron_a_pos =~ s/\d//; # e.g convert IH0 to IH.  Only affects
                  $pron_b_pos =~ s/\d//; # whether we exit the loop below.
                }
                if ($pron_a_pos ne $pron_b_pos) {
                  # This is important: we don't consider a pron suffix-pair to be
                  # valid unless the pron prefix is the same.
                  last;
                }
              }
            }
          }
        }
      }
    }
  }
  # To save memory, only keep pairs above threshold in the hash.
  while ( my ($quadruple, $count) = each (%quadruple_count) ) {
    if ($count >= $min_suffix_count) {
      $new_hash{$quadruple} = $count;
    }
  }
  %quadruple_count = %new_hash;
  undef %new_hash;
  
  # Print out the quadruples for diagnostics.
  foreach $quadruple ( 
    sort { $quadruple_count{$b} <=> $quadruple_count{$a} } keys %quadruple_count ) {
    print STDERR "$quadruple_count{$quadruple} $quadruple\n";
  }
}
# Now print out the quadruples; these are the output of this program.
foreach $quadruple (keys %quadruple_count) {
  print $quadruple."\n";
}
