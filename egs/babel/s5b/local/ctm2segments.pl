#!/usr/bin/env perl

use strict;
use warnings;
use Getopt::Long;
use Data::Dumper;

sub uniq {
    my %seen = ();
    my @r = ();
    foreach my $a (@_) {
        unless ($seen{$a}) {
            push @r, $a;
            $seen{$a} = 1;
        }
    }
    return @r;
}

sub avg_cf_mode {
  my $utt_words= shift @_;
  my $utt_cf = shift @_;
  my $utt_wdur = shift @_;

  my $counter=0.0;
  my $avg=0.0;
  foreach my $cf ( @{$utt_cf} ) {
     $avg = $avg * $counter/($counter + 1.0) + $cf/($counter + 1.0);
     $counter += 1.0;
  }
  return $avg;
}

sub min_cf_mode {
  my $utt_words= shift @_;
  my $utt_cf = shift @_;
  my $utt_wdur = shift @_;

  my $counter=0.0;
  my $min=2;
  foreach my $cf ( @{$utt_cf} ) {
     $min = $cf if ( $min > $cf ) ;
     $counter += 1.0;
  }
  return $min;
}

sub max_cf_mode {
  my $utt_words= shift @_;
  my $utt_cf = shift @_;
  my $utt_wdur = shift @_;

  my $counter=0.0;
  my $max=-1;
  foreach my $cf ( @{$utt_cf} ) {
     $max = $cf if ( $max < $cf ) ;
     $counter += 1.0;
  }
  return $max;
}


my $cf_min = 0.9;
my $cf_max = 1;
my $cf_rule = "avg" ;

my $Usage = <<EOU;
Generate kaldi text and segments file (for use during unsupervised training)
Usage:    ctm_to_training.pl [options] <ctm_in|-> <text_out|->

Allowed options:
  --min-cf           : Minimum CF to include the word (float, default = 0.9)
  --max-cf           : Maximum CF to include the word (float, default = 1.0)
  --cf-rule          : Mode to compute the per-segment CF(string, min,max,avg, default="avg")
EOU

print join(" ", $0, @ARGV) . "\n";

GetOptions('min-cf=f'   => \$cf_min,
           'max-cf=f'   => \$cf_max,
           'cf-rule=s'    => \$cf_rule,
           );


print "CF-MIN:  $cf_min\n";
print "CF-MAX:  $cf_max\n";
print "CF-RULE: $cf_rule\n";

my $decide;
if ($cf_rule eq "avg") {
  $decide = \&avg_cf_mode;
} elsif ($cf_rule eq "min" ) {
  $decide = \&min_cf_mode; 
} elsif ($cf_rule eq "max" ) {
  $decide = \&max_cf_mode; 
} else {
  die "Unknow CF rule $cf_rule\n";
}

die "Unsupported number of arguments ($#ARGV)!" if ($#ARGV != 2) ;
# Get parameters
my $data = shift @ARGV;
my $filein = shift @ARGV;
my $dirout = shift @ARGV;


my @utt_words;
my @utt_cf;
my @utt_wdur;

my $utt_name = "";
my $utt_chan = "";

my $total_seconds=0;
my $used_seconds=0;
my $used_segments=0;

my %segments;
my %segments_dur;
my %speakers;
open(INPUT_SEGMENTS, "< $data/segments") or die "Could not open file $data/segments\n";
while(<INPUT_SEGMENTS>) {
  chomp ; 
  my @entries = split / /, $_; 
  my $utt = $entries[0];
  my $dur = $entries[$#entries] - $entries[$#entries - 1];
  $segments{$utt}=$_;
  $segments_dur{$utt} = $dur;
  $total_seconds += $dur;
}
open(INPUT_SPEAKERS, "< $data/utt2spk") or die "Could not open file $data/utt2spk\n";

while(<INPUT_SPEAKERS>) {
  chomp ; 
  (my $utt, my $spk) = split / /, $_; 
  $speakers{$utt}=$spk;
}
my $hours=sprintf("%0.2f", $total_seconds/3600.0);
print "$0: Read " . (scalar keys %segments) .  " segments ($hours hours of audio)...\n";
print "$0: Read " . (scalar uniq(values %speakers) ) .  " speakers...\n";


open(OUTPUT_SEGMENTS, "|sort -u > $dirout/segments") or die;
open(OUTPUT_TEXT, "|sort -u > $dirout/text") or die;
open(OUTPUT_SPK, "|sort -u > $dirout/utt2spk") or die;

my $segment;
my $segments_read;

open(CTM, " < $filein");
while (my $ctm_line = <CTM> ) {
  chomp $ctm_line;
  my @entries = split / /, $ctm_line;
  die "$0: Could not parse line \"$ctm_line\".\n\tLine does not parse to 6 segments (parses to $#entries).\n" if $#entries != 5;
  (my $utt, my $chan_id, my $word_offset, my $word_dur, my $word, my $word_cf) = @entries ;
  
  if ($utt ne $utt_name ) {
    if ( $utt_name)  {
      my $dec = $decide->(\@utt_words, \@utt_cf, \@utt_wdur);

      if (($dec >= $cf_min) and ($dec <= $cf_max)){
        #print "$utt_name = $dec\n";
        print OUTPUT_SEGMENTS $segments{$utt_name} . "\n";
        print OUTPUT_TEXT "$utt_name " . join(" ", @utt_words) . "\n";
        print OUTPUT_SPK  "$utt_name " .$speakers{$utt_name} . "\n";
        $used_seconds += $segments_dur{$utt_name};
        $used_segments += 1;
      }
    }

    $utt_name = $utt;
    $utt_chan = $chan_id;
  
    @utt_words = ();
    @utt_cf = ();
    @utt_wdur = ();
  } else {
    push @utt_words, $word;
    push @utt_cf, $word_cf;
    push @utt_wdur, $word_dur;
  }
}
if ( $utt_name)  {
  my $dec = $decide->(\@utt_words, \@utt_cf, \@utt_wdur);

  if (($dec >= $cf_min) and ($dec <= $cf_max)){
    #print "$utt_name = $dec\n";
    print OUTPUT_SEGMENTS $segments{$utt_name} . "\n";
    print OUTPUT_TEXT "$utt_name " . join(" ", @utt_words) . "\n";
    print OUTPUT_SPK  "$utt_name " .$speakers{$utt_name} . "\n";
    $used_seconds += $segments_dur{$utt_name};
    $used_segments += 1;
  }
}

if ($used_seconds < 60) {
  $hours=sprintf("%0.2f", $used_seconds);
  print "$0: Used " . $used_segments .  " segments ($hours seconds of audio)...\n";
} elsif ($used_segments < 1800) {
  $hours=sprintf("%0.2f", $used_seconds/60.0);
  print "$0: Used " . $used_segments .  " segments ($hours minutes of audio)...\n";
} else {
  $hours=sprintf("%0.2f", $used_seconds/3600.0);
  print "$0: Used " . $used_segments .  " segments ($hours hours of audio)...\n";
}
