#!/usr/bin/perl

@ARGV <= 1 or die "Usage: quantize_vector.pl [threshold]";

my $t = 0.5;

if (scalar @ARGV == 1) {
  $t = $ARGV[0];
}

while (<STDIN>) {
  chomp;
  my @F = split;

  my $str = "$F[0]";
  for (my $i = 2; $i < $#F; $i++) {
    if ($F[$i] >= $t) {
      $str = "$str 1";
    } else {
      $str = "$str 0";
    }
  }

  print ("$str\n");
}
