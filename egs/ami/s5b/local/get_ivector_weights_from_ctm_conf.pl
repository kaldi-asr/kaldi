#! /usr/bin/perl
use strict;
use warnings;
use Getopt::Long;

my $pad_frames = 0;
my $silence_weight = 0.00001; 
my $scale_weights_by_ctm_conf = "false";
my $frame_shift = 0.01;

GetOptions('pad-frames:i' => \$pad_frames, 
           'silence-weight:f' => \$frame_shift,
           'scale-weights-by-ctm-conf:s' => \$scale_weights_by_ctm_conf,
           'frame-shift:f' => \$frame_shift);

if (scalar @ARGV != 1) {
  die "Usage: get_ivector_weights_from_ctm_conf.pl <utt2dur> < <ctm> > <out-weights>";
}

my $utt2dur = shift @ARGV;

$pad_frames >= 0 || die "Bad pad-frames value $pad_frames; must be >= 0";
($scale_weights_by_ctm_conf eq 'false') || ($scale_weights_by_ctm_conf eq 'true') || die "Bad scale-weights-by-ctm-conf $scale_weights_by_ctm_conf; must be true/false";

open(L, "<$utt2dur") || die "unable to open utt2dur file $utt2dur";

my @all_utts = ();
my %utt2weights;

while (<L>) {
  chomp;
  my @A = split;
  @A == 2 || die "Incorrent format of utt2dur file $_";
  my ($utt, $len) = @A;

  push @all_utts, $utt;
  $len = int($len / $frame_shift);
  
  # Initialize weights for each utterance
  my $weights = [];
  for (my $n = 0; $n < $len; $n++) { 
    push @$weights, $silence_weight;
  }
  $utt2weights{$utt} = $weights;
}
close(L);

while (<STDIN>) {
  chomp;
  my @A = split;
  @A == 6 || die "bad ctm line $_";

  my $utt = $A[0]; 
  my $beg = $A[2]; 
  my $len = $A[3];
  my $beg_int = int($beg / $frame_shift) - $pad_frames; 
  my $len_int = int($len / $frame_shift) + 2*$pad_frames;
  my $conf = $A[5];

  my $array_ref = $utt2weights{$utt};
  defined $array_ref || die "No length info for utterance $utt";

  for (my $t = $beg_int; $t < $beg_int + $len_int; $t++) {
    if ($t >= 0 && $t < @$array_ref) {
      if ($scale_weights_by_ctm_conf eq "false") {
        ${$array_ref}[$t] = 1;
      } else {
        ${$array_ref}[$t] = $conf;
      }
    }
  }
}

foreach my $utt (keys %utt2weights) {
  my $array_ref = $utt2weights{$utt};
  print ($utt, " [ ", join(" ", @$array_ref), " ]\n");
}
