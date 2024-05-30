#!/usr/bin/env perl
# Copyright 2017     Hossein Hadian
#           2017     Aswin Shanmugam Subramanian

$randseed = 0;
$shifted = 0;

do {
  $shifted=0;
  if ($ARGV[0] eq "--first-column") {
    $first_column = $ARGV[1];
    shift @ARGV; shift @ARGV;
    $shifted = 1;
  }
  if ($ARGV[0] eq "--first-element") {
    $first_element = $ARGV[1];
    shift @ARGV; shift @ARGV;
    $shifted = 1;
  }
  if ($ARGV[0] eq "--srand") {
    $randseed = $ARGV[1];
    shift @ARGV; shift @ARGV;
    $shifted = 1;
  }
  if ($ARGV[0] eq "--stddev") {
    $stddev = $ARGV[1];
    shift @ARGV; shift @ARGV;
    $shifted = 1;
  }
} while ($shifted);

if(@ARGV != 2) {
  die "Usage: $0 [--srand <int>] [--first-column <float>] [--first-element <float>]" .
    "[--stddev <float>] <num-rows> <num-cols> > matrix.txt\n" .
    "This script is used to randomly initialize \n" .
    "Kaldi-format matrices, with supplied dimensions and \n" .
    "standard deviation.\n"
}
($num_rows, $num_cols) = @ARGV;


$stddev = 1.0 / sqrt($num_cols) if not defined $stddev;
srand($randseed);
my $pi2 = 6.2831853;

print "[ ";
for ($i = 0; $i < $num_rows; $i++) {
  for ($j = 0; $j < $num_cols; $j++) {
    $flag = 0;
    if ($j == 0 && defined $first_column) {
      $element1 = $first_column;
      $flag = 1;
    } elsif ($i == 0 && $j == 0 && defined $first_element) {
      $element1 = $first_element;
      $flag = 1;
    } elsif ($stddev == 0) {
      $element1 = 0;
      $element2 = 0;
    } else {
      $a = rand();
      $b = rand();
      $u1 = sqrt(-2 * log($a));
      $u2 = $pi2*$b;
      $element1 = $u1 * cos($u2);
      $element2 = $u1 * sin($u2);
      $element1 = $element1 * $stddev;
      $element2 = $element2 * $stddev;
    }
    printf "%0.2f ", $element1;
    if ($j != $num_cols - 1 && $flag != 1) {
      printf "%0.2f ", $element2;
      $j = $j + 1;
    }
  }
  print "\n";
}
print "]\n";
