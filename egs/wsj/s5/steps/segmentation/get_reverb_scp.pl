#! /usr/bin/perl
use strict;
use warnings;

my $field_begin = -1;
my $field_end = -1;

if ($ARGV[0] eq "-f") {
  shift @ARGV; 
  my $field_spec = shift @ARGV; 
  if ($field_spec =~ m/^\d+$/) {
    $field_begin = $field_spec - 1; $field_end = $field_spec - 1;
  }
  if ($field_spec =~ m/^(\d*)[-:](\d*)/) { # accept e.g. 1:10 as a courtesty (properly, 1-10)
    if ($1 ne "") {
      $field_begin = $1 - 1; # Change to zero-based indexing.
    }
    if ($2 ne "") {
      $field_end = $2 - 1; # Change to zero-based indexing.
    }
  }
  if (!defined $field_begin && !defined $field_end) {
    die "Bad argument to -f option: $field_spec"; 
  }
}

if (scalar @ARGV != 1 && scalar @ARGV != 2 ) {
  print "Usage: get_reverb_scp.pl [-f <field-start>-<field-end>] <num-reps> [<prefix>] < input_scp > output_scp\n";
  exit(1);
}

my $num_reps = $ARGV[0];
my $prefix = "rev";

if (scalar @ARGV == 2) {
    $prefix = $ARGV[1];
}

while (<STDIN>) {
  chomp;
  my @A = split;

  for (my $i = 1; $i <= $num_reps; $i++) {
    for (my $pos = 0; $pos <= $#A; $pos++) {
      my $a = $A[$pos];
      if ( ($field_begin < 0 || $pos >= $field_begin)
        && ($field_end < 0 || $pos <= $field_end) ) {
        if ($a =~ m/^(sp[0-9.]+-)(.+)$/) {
          $a = $1 . "$prefix" . $i . "_" . $2;
        } else {
          $a = "$prefix" . $i . "_" . $a;
        }
      }
      print $a . " ";
    }
    print "\n";
  }
}
