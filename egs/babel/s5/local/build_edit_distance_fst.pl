#!/usr/bin/perl

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)
# Apache 2.0.
#

use strict;
use warnings;
use Getopt::Long;

my $Usage = <<EOU;
Usage:    build_edit_distance_fst.pl <phones.txt|-> <fst_out|->
          Buld a edit distance FST at the phone level.

Allowed options:
  --confusion-matrix    : Matrix for insertion, deletion and substitution. (string, default="")
  --ins-cost            : Insertion cost                                   (double, default=1 )
  --del-cost            : Deletion cost                                    (double, default=1 )
  --subs-cost           : substitution cost                                (double, default=1 )
  --boundary-ins-cost   : Cost for insertions at work boundary             (double, default=0.1)
  --boundary-off        : No insertions at word boundary                   (boolean, default=true)
EOU

my $confusion_matrix = "";
my $insertion_cost = 1;
my $deletion_cost = 1;
my $substitution_cost = 1;
my $boundary_ins_cost = 0.1;
my $boundary_off="true";
GetOptions('confusion-matrix=s' => \$confusion_matrix,
  'ins-cost=f'          => \$insertion_cost,
  'del-cost=f'          => \$deletion_cost,
  'subs-cost=f'         => \$substitution_cost,
  'boundary-ins-cost=f' => \$boundary_ins_cost,
  'boundary-off=s'      => \$boundary_off);

@ARGV == 2 || die $Usage;

$boundary_off eq "true" || $boundary_off eq "false" || die "$0: Bad value for option --boundary-off\n";

# Workout the input and output parameters
my $phone_in = shift @ARGV;
my $fst_out = shift @ARGV;

open(I, "<$phone_in") || die "$0: Fail to open lexicon $phone_in\n";
open(O, ">$fst_out") || die "$0: Fail to write FST $fst_out\n";

# Read confusion matrix
my %confusion;
if ($confusion_matrix ne "") {
  open(M, "<$confusion_matrix") || die "$0: Fail to open confusion matrix $confusion_matrix\n";
  while (<M>) {
    chomp;
    my @col = split();
    @col == 3 || die "$0: Bad line in confusion matrix \"$_\"\n";
    $confusion{"$col[0]_$col[1]"} = $col[2];
  }
  close(M);
}

# Start processing
my @phones;
while (<I>) {
  chomp;
  my @col = split();
  @col == 1 || die "$0: Bad number of columns in phone list \"$_\"\n";
  if ($col[0] eq "<eps>") {next;}
  push(@phones, $col[0]);
}

# Add insertions, deletions
my $fst = "";
foreach my $p (@phones) {
  if ($confusion_matrix eq "") {
    $fst .= "1 1 $p <eps> $deletion_cost\n";        # Deletions
    $fst .= "1 1 <eps> $p $insertion_cost\n";       # Insertions
    if ($boundary_off eq "false") {
      $fst .= "0 0 <eps> $p $boundary_ins_cost\n";
      $fst .= "0 1 <eps> $p $boundary_ins_cost\n";
      $fst .= "2 2 <eps> $p $boundary_ins_cost\n";
      $fst .= "1 2 <eps> $p $boundary_ins_cost\n";
    }
  } else {
    my $key = "${p}_<eps>";
    if (defined($confusion{$key})) {
      $fst .= "1 1 $p <eps> $confusion{$key}\n";
    }
    $key = "<eps>_${p}";
    if (defined($confusion{$key})) {
      $fst .= "1 1 <eps> $p $confusion{$key}\n";
      if ($boundary_off eq "false") {
        $fst .= "0 0 <eps> $p $confusion{$key}\n";
        $fst .= "0 1 <eps> $p $confusion{$key}\n";
        $fst .= "2 2 <eps> $p $confusion{$key}\n";
        $fst .= "1 2 <eps> $p $confusion{$key}\n";
      }
    }
  }
}
foreach my $p1 (@phones) {
  foreach my $p2 (@phones) {
    if ($p1 eq $p2) {
      $fst .= "1 1 $p1 $p2 0\n";
    } else {
      if ($confusion_matrix eq "") {
        $fst .= "1 1 $p1 $p2 $substitution_cost\n";
      } else {
        my $key = "${p1}_${p2}";
        if (defined($confusion{$key})) {
          $fst .= "1 1 $p1 $p2 $confusion{$key}\n";
        }
      }
    }
  }
}
if ($boundary_off eq "false") {
  $fst .= "0 1 <eps> <eps> 0\n";
  $fst .= "1 2 <eps> <eps> 0\n";
  $fst .= "2\n";
} else {
  $fst .= "1\n";
}

print O $fst;

close(I);
close(O);
