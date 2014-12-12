#!/usr/bin/perl
# Apache 2.0.
# special-purpose script to make un-smoothed phone
# bigram LM.  Input looks like:
# 1 292 290 346 178 185 1 232 245 312 1 
# 1 216 42 18 38 346 145 244 177 247 228 206 146 246 18 37 256 314 206 174 1 


$max_phone = 0;
while (<>) { 
  # note, don't allow initial-to-final transition.
  @A = split(" ", $_);
  if (@A == 0) { next; }
  $tot_initial_count++;
  $initial_count[$A[0]]++;
  for ($n = 0; $n < @A; $n++) {
    $p = $A[$n];
    $p > 0 || die "bad phone $p\n"; # Check it's an integer more than 0.
    if ($p > $max_phone) { $max_phone = $p; } 
    $tot_count[$p]++; # denominator of probabilities.
    if ($n+1 == @A) {
      $final_count[$p]++;
    } else {
      $q = $A[$n+1];
      $p > 0 || die "bad phone $q\n"; # Check it's an integer more than 0.
      $bigram_count{$p,$q}++;
    }
  }
}

# Will have a state for each of the phones, with the
# same numbering; state 0 is initial state. state $end_state is final state.
$end_state = $max_phone+1;
for ($p = 1; $p <= $max_phone; $p++) {
  if ($initial_count[$p] != 0.0) {
    $cost = -log($initial_count[$p] / $tot_initial_count);
    # format is: from-state to-state input-symbol output-symbol cost.
    if ($cost == 0.0) { $cost = "0"; } # otherwise was -0.
    print "0  $p  $p  $p  $cost\n";

  }
}

# Now bigram transitions
foreach $k (keys %bigram_count) {
  ($p,$q) = split($;, $k);
  ($p > 0 && $q > 0) || die;
  $cost = -log($bigram_count{$k} / $tot_count[$p]);
  if ($cost == 0.0) { $cost = "0"; } # otherwise was -0.
  # format is: from-state to-state input-symbol output-symbol cost.
  print "$p  $q  $q  $q  $cost\n";
}

# Now final costs.
for ($p = 1; $p <= $max_phone; $p++) {
  if ($final_count[$p] != 0.0) {
    $cost = -log($final_count[$p] / $tot_count[$p]);
    if ($cost == 0.0) { $cost = "0"; } # otherwise was -0.
    # Format is: final-state final-cost.
    print "$p  $cost\n";
  }
}

__END__

echo 1 2 | ./make_phone_bigram.pl
