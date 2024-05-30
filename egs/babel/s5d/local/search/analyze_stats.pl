#!/usr/bin/env perl
#===============================================================================
# Copyright 2015  (Author: Yenda Trmal <jtrmal@gmail.com>)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.
#===============================================================================


my $Usage = <<EOU;
Takes the search output (alignment.csv and the statistics) and for each keyword
it tries to work out if the given path increases the ATWV or decreases it.
Those which decrease the ATWV can be subsequently removed from the keyword FST.

Usage: cat stats | $0 [options] <datadir> <kws-alignment> <output-stats>
 e.g.: gunzip -c exp/tri5/decode_dev10h.pem/kws/stats.*.gz | \
         $0 --trials 36000 data/dev10h.pem alignment.csv keywords_stats

Allowed options:
  --trials  : number of trials (length of the search collection) for ATWV computation
EOU

use strict;
use warnings;
use utf8;
use Data::Dumper;
use GetOpt::Long;

my $T = 36212.6725;

GetOptions ("trials=i" => \$T) or do
 {
  print STDERR "Cannot parse the command-line parameters.\n";
  print STDERR "$Usage\n";
  die "Cannot continue\n"
}

if (@ARGV != 3) {
  print STDERR "Incorrect number of command-line parameters\n";
  print STDERR "$Usage\n";
  die "Cannot continue\n"
}

my $data = $ARGV[0];
my $align = $ARGV[1];
my $keywords = $ARGV[2];

my %SEGMENTS;
open(my $seg_file, "$data/segments") or
  die "Cannot open the segments file in $data/segments";

while (my $line = <$seg_file>) {
  (my $seg_id, my $file_id, my $tstart, my $tend) = split(" ", $line);
  $SEGMENTS{$seg_id} = [$file_id, $tstart, $tend];
}


my %ALIGNMENT;
my %TWVSTATS;
open(my $align_file, $align) or
  die "Cannot open the alignment file in $align";

print "Reading alignment...\n";
my $dummy=<$align_file>;
while (my $line = <$align_file>) {
  chomp $line;
  my @entries = split(/\s*,\s*/, $line);
  my $kw_id = $entries[3];
  my $file_id = $entries[1];
  my $kw_time = $entries[7];
  my $op_id = join(",", @entries[10 .. 11]); # 'YES,CORR' | 'YES,FA' | 'NO,MISS' | 'NO,CORR!DET' | ',MISS'

  $TWVSTATS{$kw_id}{$op_id} += 1;
  next if $op_id eq ",MISS";

  my $key = sprintf "%s,%s", $kw_id, $file_id;

  if ( grep { abs($_ -  $kw_time) <= 0.5 } @{$ALIGNMENT{$key}} ) {
      die "The key $key is not unique\n";
  }
  push @{$ALIGNMENT{$key}}, \@entries;
}

#print Dumper(\%TWVSTATS);
print "Done reading alignment...\n";


my %HITCACHE;

print "Reading stats\n";
while (my $line = <STDIN> ) {
  my @entries = split(" ", $line);

  my $wav = $SEGMENTS{$entries[1]}[0];
  my $seg_start = $SEGMENTS{$entries[1]}[1];
  my $seg_end = $SEGMENTS{$entries[1]}[2];

  my $kw = $entries[0];
  my $kw_start = $seg_start  + $entries[2]/100.00000;
  my $kw_stop = $seg_start  + $entries[3]/100.00000;
  my $kw_center = ($kw_start + $kw_stop) / 2.0;
  #print Dumper($kw_start, $kw_stop, $kw_center);
  my $kw_wav = $wav;

  my $key = sprintf "%s,%s", $kw, $kw_wav;

  if ( not grep { abs( (@{$_}[7] + @{$_}[8])/2.0 -  $kw_center) <= 0.1 } @{$ALIGNMENT{$key}} ) {
      ##print  "The key $key, $kw_center does not exist in the alignment\n";
      ##print join(" ", @entries) . "\n";
      #print Dumper($ALIGNMENT{$key});
      #die;
  } else {
      my @tmp = @{$ALIGNMENT{$key}};
      my ($index) = grep { abs(  (@{$tmp[$_]}[7] + @{$tmp[$_]}[8]) / 2.0 - $kw_center) <= 0.1 } (0 .. @{$ALIGNMENT{$key}}-1);
      die unless defined $index;
      my @ali = @{@{$ALIGNMENT{$key}}[$index]};
      my $diff = abs($ali[7] - $kw_start);

      #die "Weird hit " . Dumper(\@entries) if $entries[5] != 0;

      my $hit_id = join(" ", @entries[5 .. @entries-1]);
      $hit_id =~ s/\b0\b//g;
      $hit_id =~ s/^\s+//g;
      $hit_id =~ s/\s+/ /g;
      $hit_id =~ s/\s+$//g;
      #print $hit_id . "\n";
      #print Dumper(\@ali, $kw_wav, $diff) if $diff > 0.1;
      #print Dumper(\@entries);

      my $op_id = join(",", @ali[10 .. 11]); # 'YES,CORR' | 'YES,FA' | 'NO,MISS' | 'NO,CORR!DET'
      $HITCACHE{$kw}{$hit_id}{$op_id} += 1;
      #push @{$HITCACHE{$hit_id}{join(",", @ali[10 .. 11])}}, $entries[4];
  }
  #print Dumper(\@entries, $kw_start, $kw_wav);
  #exit
}
#print Dumper(\%HITCACHE);
print "Done reading stats\n";

open(my $KW, "> $keywords");

print "Analyzing\n";
my $TWV = 0;
my $NEW_TWV = 0;
my $N_KW = 0;
foreach my $kwid (sort keys %HITCACHE) {
  my %old_stats = %{$TWVSTATS{$kwid}};
  #print Dumper($kwid, \%old_stats);
  #
  $old_stats{"YES,CORR"} = 0 unless defined $old_stats{"YES,CORR"};
  $old_stats{",MISS"} = 0 unless defined $old_stats{",MISS"};
  $old_stats{"NO,MISS"} = 0 unless defined $old_stats{"NO,MISS"};
  $old_stats{"YES,FA"} = 0 unless defined $old_stats{"YES,FA"};

  my $n_kw = $old_stats{"YES,CORR"} +
             $old_stats{",MISS"} +
             $old_stats{"NO,MISS"};

  my $n_trials = $T - $n_kw;

  next if $n_kw == 0;

  my $p_miss = 0;
  $p_miss = 1 -  $old_stats{"YES,CORR"} / $n_kw unless $n_kw == 0;
  my $p_fa = $old_stats{"YES,FA"} / $n_trials;

  my $twv = 1 - $p_miss - 999.9 * $p_fa;
  print "$kwid $n_kw $p_miss $p_fa $twv\n";

  foreach my $kwpath (sort keys $HITCACHE{$kwid}) {
    my $weight = 0;

    my %new_stats =  %{$HITCACHE{$kwid}{$kwpath}};
    $new_stats{"YES,CORR"} = 0 unless defined $new_stats{"YES,CORR"};
    $new_stats{"YES,FA"} = 0 unless defined $new_stats{"YES,FA"};

    my $new_p_miss = 1 -  ($old_stats{"YES,CORR"}  - $new_stats{"YES,CORR"})/ $n_kw;
    my $new_p_fa = ($old_stats{"YES,FA"}  - $new_stats{"YES,FA"}) / $n_trials;
    my $new_twv = 1 - $new_p_miss - 999.9 * $new_p_fa;
    if ($new_twv > $twv) {
      #print "keep: $kwid $kwpath $twv - $new_twv\n";
      if ((defined $HITCACHE{$kwid}{$kwpath}->{"YES,FA"}) ||
          (defined $HITCACHE{$kwid}{$kwpath}->{"NO,MISS"}) ||
          (defined $HITCACHE{$kwid}{$kwpath}->{"YES,CORR"})) {
        print Dumper($kwid, $kwpath, $HITCACHE{$kwid}{$kwpath});
      }
      $old_stats{"YES,CORR"} -= $new_stats{"YES,CORR"};
      $old_stats{"YES,FA"} -= $new_stats{"YES,FA"} ;
    } else {
      print $KW "$kwid $kwpath\n";
      #print "remove: $kwid $kwpath $twv - $new_twv\n";

    }
    # print $W "$kwid $weight\n";

  }


  my $new_p_miss = 1 -  $old_stats{"YES,CORR"} / $n_kw;
  my $new_p_fa = $old_stats{"YES,FA"} / $n_trials;

  my $new_twv = 1 - $new_p_miss - 999.9 * $new_p_fa;

  $NEW_TWV = $N_KW/($N_KW+1) * $NEW_TWV + $new_twv / ($N_KW+1);
  $TWV = $N_KW/($N_KW+1) * $TWV + $twv / ($N_KW+1);
  $N_KW += 1;
}
close($KW);
#print "ATWV: $TWV $NEW_TWV\n";
