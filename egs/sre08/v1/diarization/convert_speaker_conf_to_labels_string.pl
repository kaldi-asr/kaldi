#!/usr/bin/perl 

use strict;
use POSIX;
use Pod::Usage;
use Getopt::Long;

my $help = 0;
my $min_spk_conf = 0.8;
my $skip_empty_label_recos = "true";

GetOptions('min-spk-conf:f' => \$min_spk_conf,
           'help|?' => \$help,
           'skip-empty-label-recos:s' => \$skip_empty_label_recos);

if ((@ARGV > 1 || $help)) {
  print STDERR "$0:\n" .
               "Usage: convert_speaker_conf_to_labels.pl [options] [speaker-conf] > <remove-labels-csl-archive>\n";
  exit 0 if $help;
  exit 1;
}

# This is how a line of speaker-conf can look like:
# <reco-id> [ (<spk-id> <confidence>)+ ] [ (<spk-id> <occupancy>)+ ]
#single_f1e7fa29-1 [ 0 0.7433829 1 0.9339832 2 0.9105379 ] [ 0 413.2838 1 818.2226 2 70.9536 ]

($min_spk_conf > 0.0 && $min_spk_conf <= 1.0) or die "min-spk-conf must be between 0 and 1";

while (<>) {
  chomp;

  (m/^\s*(\S+)\s+\[((?:\s+[0-9.+]+\s+[0-9.+]+)+)\s+\]\s+\[((?:\s+[0-9.+]+\s+[0-9.+]+)+)\s+\]/) or die "Unparsable line $_ in speaker_conf file";

  print STDERR "Parsed line: $1 $2 , $3\n";
  my $reco = $1;
  my @A = split(' ', $2);
  my @B = split(' ', $3);

  ($#A % 2 == 1) or die "Bad line $_: speaker confidences must be of the format <spk-id> <conf>";
  $#A == $#B or die "Bad line $_";

  my @remove_labels;
  my @occupancies;

  my $avg_conf = 0;
  my $tot_occ = 0;

  for (my $i = 0; $i < @A; $i = $i + 2) {
    $avg_conf += $A[$i+1] * $B[$i+1];
    $tot_occ += $B[$i+1];
  }

  $avg_conf /= $tot_occ;

  for (my $i = 0; $i < @A; $i = $i + 2) {
    print STDERR "Extracted label : $A[$i], $A[$i+1] " . $min_spk_conf * $avg_conf . "\n";
    if ($A[$i+1] < $min_spk_conf * $avg_conf) {
      push @remove_labels, $A[$i];
    }
  }

  if (($skip_empty_label_recos eq "true") && ( scalar @A == 0 )) {
    print STDOUT $reco . " " . join(':', @remove_labels) . "\n";
  }
}
