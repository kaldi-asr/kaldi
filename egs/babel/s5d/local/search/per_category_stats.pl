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

# Takes the alignment.csv and the category tables and computes the per-category
# statistics including the oracle measures (OTWV, MTWV, STWV)
# Is not particulary effective (for example, it computes the oracle measures
# for each keyword several times (once for each category the keyword is in);
# To achieve at least partial speed-up, we cache some of the partial statistics
# The caching gave us speed improvement approx. from 22s down to 14s
#
# The lines in output starting with '#' are intended as comments only -- you
# can filter them out using grep -v '^#'
# The first comment line contains header,
# The second cooment line contains column numbers (to make easier using cut -f)
#   -- you don't have to count the fields, just use the present
#      number of the field
#
# Compatibility:
#   We tried to make the numbers comparable with F4DE output. If there is a large
#   difference, something is probably wrong and you should report it
#   The column names should be compatible (to large extent) with F4DE output
#   files (sum.txt, bsum.txt, cond.bsum.txt). Our intention was, however,
#   to make this file easily grepable/machine-processable, so we didn't honor
#   the original F4DE file fomrat
#
# Usage:
#   It reads the alignment.csv from the STDIN.
#   Moreover, it expects exactly two arguments: number of trials and
#   the category table
# I.e.
#   local/search/per_category_stats.pl <trials> <categories>
#
# Example:
#   cat alignment.csv | perl local/search/per_category_stats.pl  `cat data/dev10h.pem/extra_kws/trials` data/dev10h.pem/extra_kws/categories
#
# Additional parameters
#   --beta           # beta value (weight of FAs), defailt 999.9
#   --sweep-step     # sweep step for the oracle measures
#
# TODO
#   Document what each field means (might be slightly tricky, as even F4DE
#   does not document the exact meaning of some of the fields.
#
#   ATWV - actual Term-Weighted Value (TWV for the threshold 0.5)
#   MTWV - Maximum Term-Weighted Value (TWV for the threshold that maximizes
#          the given category's TWV
#   OTWV - Optimum Term-Weighted Value (TWV assuming the decision threshold
#          for each Term/KW is determined optimally)
#   STWV - Supreme TWV - essentially Lattice Recall

use strict;
use warnings FATAL => 'all';
use utf8;
use List::Util;
use Data::Dumper;
use Getopt::Long;
use Scalar::Util qw(looks_like_number);

binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";
binmode STDERR, ":utf8";

my %CATEGORIES;
my %STATS;
my %K;

my $beta=999.9;
my $step_size=0.005;
my $threshold = 0.5;
my $enable_caching = 1;

my $cat_maxlen = 9; #Must accomodate string "#CATEGORY" in the header
my $field_size = 9;

my $L = int(1.0/$step_size) + 1;

GetOptions("beta=f" => \$beta,
           "sweep-step=f" => \$step_size,
           "disable-caching" => sub{ $enable_caching=''; }
          ) or die "Cannot process the input options (possibly unknown switch)";

die "Unsupported number of arguments." if @ARGV != 2;
if ( not looks_like_number($ARGV[0])) {
  die "The first parameter must be a float number (number of trials) -- got $ARGV[0]";
}

my $T= 0.0 + $ARGV[0];


open(CAT, $ARGV[1]) or die("Cannot open categories file $ARGV[1]");
while(my $line = <CAT>) {
  my @entries =split(" ", $line);

  die "Unknown format of category line: \"$line\"" if scalar @entries < 2;
  my $kw = shift @entries;


  if (not defined $STATS{$kw}->{fa_sweep}) {
    $STATS{$kw}->{fa} = 0;
    $STATS{$kw}->{corr} = 0;
    $STATS{$kw}->{miss} = 0;
    $STATS{$kw}->{lattice_miss} = 0;
    $STATS{$kw}->{ntrue} = 0;
    $STATS{$kw}->{count} = 0;
    $STATS{$kw}->{corrndet} = 0;

    my @tmp1 = (0) x ($L+1);
    $STATS{$kw}->{fa_sweep} = \@tmp1;
    my @tmp2 = (0) x ($L+1);
    $STATS{$kw}->{corr_sweep} = \@tmp2;
  }

  push @entries, "ALL";
  foreach my $cat (@entries) {
    $cat_maxlen = length($cat) if length($cat) > $cat_maxlen;
    push @{$CATEGORIES{$cat}}, $kw;
    $K{$cat} += 1;
  }
}
close(CAT);
#print Dumper(\%CATEGORIES);


#print STDERR "Reading the whole CSV\n";
my $i = 0;
my $dummy=<STDIN>;
while (my $line=<STDIN>) {
  chomp $line;
  my @entries = split(",", $line);

  die "Unknown format of category line: \"$line\"" if scalar @entries != 12;


  my $termid = $entries[3];
  my $ref_time = $entries[5];
  my $score = $entries[9];
  my $decision=$entries[10];
  my $ref = $entries[11];

  if (not defined($STATS{$termid}->{ntrue})) {
    print STDERR "Term $termid not present in the category table, skipping\n";
    next
  }
  #print "$termid, ref_time=$ref_time, score=$score, start=" . int($score/$step_size + 0.5) . ", L=$L\n" if $termid eq  "KW303-00025";
  if ($score) {
    $score = 1.0 if $score > 1.0;
    my $q = int($score/$step_size) + 1;
    for (my $i = 0; $i < $q ; $i += 1) {
      if ($ref_time) {
        $STATS{$termid}->{corr_sweep}->[$i]  += 1;
      } else {
        $STATS{$termid}->{fa_sweep}->[$i]  += 1;
      }
    }
  }

  #print STDERR "$line ";
  $STATS{$termid}->{count} += 1 if $score;

  #print Dumper($ref_time, $score, $STATS{$termid}) if ($ref_time);
  if (($decision eq "YES") && ($ref eq "FA")) {
    $STATS{$termid}->{fa} += 1;
  } elsif (($decision eq "YES") && ($ref eq "CORR")) {
    $STATS{$termid}->{corr} += 1;
    $STATS{$termid}->{ntrue} += 1;
  } elsif  ($ref eq "MISS") {
    $STATS{$termid}->{lattice_miss} += 1 unless $decision;
    $STATS{$termid}->{miss} += 1;
    $STATS{$termid}->{ntrue} += 1;
  } elsif ($ref eq "CORR!DET") {
    $STATS{$termid}->{corrndet} += 1;
  }
  #print STDERR "Done\n";

}

#print STDERR "Read the whole CSV\n";

# Create the header
my $H=sprintf "%*s", $cat_maxlen-1, "CATEGORY";
my @int_vals = map{ sprintf("%*s", $field_size, $_) } (split " ", "#KW #Targ #NTarg #Sys #CorrDet #CorrNDet #FA #MISS");
my @float_vals = map{ sprintf("%*s", $field_size, $_) } (split " ", "ATWV MTWV OTWV STWV PFA MPFA OPFA PMISS MPMISS OPMISS THR MTHR OTHR");
print "#" . join(" ", $H, @int_vals, @float_vals) . "\n";
# Create secondary header with column numbers (to make cut'ing easier
my @col_nrs = map { sprintf "%*d", $field_size, $_ } (2.. 1+@int_vals + @float_vals);
print "#" . join(" ", sprintf("%*d", $cat_maxlen-1, 1),  @col_nrs) . "\n";
# End of the header

my %CACHE = ();

foreach my $cat (sort keys %CATEGORIES) {
  my $K = 0;
  my $ATWV = 0;
  my $STWV = 0;
  my $PMISS = 0;
  my $PFA = 0;

  my $OTWV = 0;
  my $OPMISS = 0;
  my $OPFA = 0;
  my $OTHR = 0;

  my $NTRUE = 0;
  my $CORR = 0;
  my $FA = 0;
  my $MISS = 0;
  my $COUNT = 0;
  my $CORRNDET = 0;

  my @MTWV_SWEEP = (0) x ($L+1);
  my @MPMISS_SWEEP = (0) x ($L+1);
  my @MPFA_SWEEP = (0) x ($L+1);
  #print Dumper($cat, $CATEGORIES{$cat});
  foreach my $kw (sort @{$CATEGORIES{$cat}}) {
    #print Dumper($kw, $STATS{$kw});
    next unless defined $STATS{$kw}->{ntrue};
    next if $STATS{$kw}->{ntrue} == 0;
    my $pmiss = 1 - $STATS{$kw}->{corr}/$STATS{$kw}->{ntrue};
    my $pfa =  $STATS{$kw}->{fa}/($T - $STATS{$kw}->{ntrue});
    my $twv = 1 - $pmiss - $beta *  $pfa;
    my $stwv = 1 - $STATS{$kw}->{lattice_miss}/$STATS{$kw}->{ntrue};

    $NTRUE += $STATS{$kw}->{ntrue};
    $CORR += $STATS{$kw}->{corr};
    $CORRNDET += $STATS{$kw}->{corrndet};
    $FA += $STATS{$kw}->{fa};
    $MISS += $STATS{$kw}->{miss};
    $COUNT += $STATS{$kw}->{count} if $STATS{$kw}->{ntrue} > 0;

    $ATWV = ($K * $ATWV + $twv) / ($K + 1);
    $PMISS = ($K * $PMISS + $pmiss) / ($K + 1);
    $PFA = ($K * $PFA + $pfa) / ($K + 1);

    $STWV = ($K * $STWV + $stwv ) / ($K + 1);

    $pmiss = 0;
    $pfa =  0;
    $twv = -99999;
    my $othr = -0.1;
    #print Dumper($kw, $STATS{$kw});
    if (($enable_caching) && (defined $CACHE{$kw})) {
      ($pfa, $pmiss, $twv, $OTHR, my $twv_sweep_cache, my $pfa_sweep_cache, my $pmiss_sweep_cache) = @{$CACHE{$kw}};
      @MTWV_SWEEP = map {($K * $MTWV_SWEEP[$_] + $twv_sweep_cache->[$_]) / ($K + 1)} (0..$L);
      @MPFA_SWEEP = map {($K * $MPFA_SWEEP[$_] + $pfa_sweep_cache->[$_]) / ($K + 1)} (0..$L);
      @MPMISS_SWEEP = map{($K * $MPMISS_SWEEP[$_] + $pmiss_sweep_cache->[$_]) / ($K + 1)} (0..$L);
    } else {
      my @twv_sweep_cache = (0) x ($L+1);
      my @pmiss_sweep_cache = (0) x ($L+1);
      my @pfa_sweep_cache = (0) x ($L+1);

      for (my $i = 0; $i <= $L; $i += 1) {
        my $sweep_pmiss = 1 - $STATS{$kw}->{corr_sweep}->[$i]/$STATS{$kw}->{ntrue};
        my $sweep_pfa =  $STATS{$kw}->{fa_sweep}->[$i]/($T - $STATS{$kw}->{ntrue});
        my $sweep_twv = 1 - $sweep_pmiss - $beta *  $sweep_pfa;
        if ($twv < $sweep_twv) {
          $pfa = $sweep_pfa;
          $pmiss = $sweep_pmiss;
          $twv = $sweep_twv;
          $OTHR = ($i - 1) * $step_size;
        }
        $pmiss_sweep_cache[$i] = $sweep_pmiss;
        $pfa_sweep_cache[$i] = $sweep_pfa;
        $twv_sweep_cache[$i] = $sweep_twv;

        #print "$i $sweep_pmiss $sweep_pfa $sweep_twv\n";
        $MTWV_SWEEP[$i] = ($K * $MTWV_SWEEP[$i] + $sweep_twv) / ($K + 1);
        $MPFA_SWEEP[$i] = ($K * $MPFA_SWEEP[$i] + $sweep_pfa) / ($K + 1);
        $MPMISS_SWEEP[$i] = ($K * $MPMISS_SWEEP[$i] + $sweep_pmiss) / ($K + 1);
      }
      $CACHE{$kw} = [$pfa, $pmiss, $twv, $OTHR, \@twv_sweep_cache, \@pfa_sweep_cache, \@pmiss_sweep_cache];
    }

    $OTWV = ($K * $OTWV + $twv) / ($K + 1);
    $OPMISS = ($K * $OPMISS + $pmiss) / ($K + 1);
    $OPFA = ($K * $OPFA + $pfa) / ($K + 1);
    $K += 1;
  }

  my $max_idx = 0;
  my $MTWV = $MTWV_SWEEP[0];
  my $MPMISS = $MPMISS_SWEEP[0];
  my $MPFA = $MPFA_SWEEP[0];
  my $MTHR = 0;
  for(my $i = 1; $i <= $L; $i += 1) {
    if ($MTWV_SWEEP[$i] > $MTWV) {
      $max_idx = $i;
      $MTWV = $MTWV_SWEEP[$i];
      $MPMISS = $MPMISS_SWEEP[$i];
      $MPFA = $MPFA_SWEEP[$i];
      $MTHR = ($i - 1) * $step_size;
    }
  }

  if ($K > 1) {
    $OTHR = "NA";
  }

  my $ntarg = $CORRNDET + $FA;

  my @abs_nrs = ($K, $NTRUE, $ntarg, $COUNT, $CORR, $CORRNDET, $FA, $MISS);
  @abs_nrs = map { sprintf "%*d", $field_size, $_ } @abs_nrs;
  my @flt_nrs = map { $_ eq "NA" ? sprintf "%6s", $_ : sprintf "% 6.3g", $_ } ($ATWV, $MTWV, $OTWV, $STWV, $PFA, $MPFA, $OPFA, $PMISS, $MPMISS, $OPMISS, 0.5, $MTHR, $OTHR);
  @flt_nrs = map {sprintf "%*s", $field_size, $_} @flt_nrs;

  my $nrs = join(" ", @abs_nrs, @flt_nrs);

  $cat = sprintf("%*s", $cat_maxlen, $cat);
  print "$cat $nrs \n";
}


