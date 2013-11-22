#!/usr/bin/perl -w
# Copyright 2012  Johns Hopkins University (Author: Jan Trmal)
# Apache 2.0.

#This script takes the original BABEL STM file (part of the IndusDB)
#and replaces the "Aggregated" field with a correct speaker ID.
#As a result, the scoring will be done on per-speaker basis as well
#As the segment from segment mapping generally do not correspond to
#the segmentation of the original STM file, it combines the files
#segments and utt2spk to work out the correct speaker ID for 
#the reference segment
#In case of overlay, it will either use the previous speaker or
#prints out an error message

use strict;
use warnings;

use Data::Dumper;

@ARGV == 2 || die "$0 <stm-file> <data-dir>\n";

my $warn_count = 0;
my $warn_max = 10;
my $stm_file = shift @ARGV;
my $data_dir = shift @ARGV;
my %utt2spk;
my %segments;

open(F_u, "<$data_dir/utt2spk") || die "Could not open the file $data_dir/utt2spk\n";
while(<F_u>) {
  chop;
  (my $utt, my $spk) = split;
  $utt2spk{$utt} = $spk;
}
close(F_u);

open(F_s, "<$data_dir/segments") || die "Could not open the file $data_dir/segments\n";
while(<F_s>) {
  chop;
  (my $utt, my $file, my $seg_start, my $seg_end) = split;
  push @{$segments{$file}}, [ $seg_start, $seg_end, $utt2spk{$utt}];
}
close(F_s);

open(STM, "<$stm_file") || die "Could not opent the STM file $stm_file";
open(STMOUT, ">$data_dir/stm") || die "Could not open the output STM file $data_dir/stm";
open(RECO, ">$data_dir/reco2file_and_channel") or die "Could not create the output file $data_dir/reco2file_and_channel";

my $prev_filename = "";
my @timestamps;
my $i = 0;
while(<STM>) {
  chop;
  (my $filename, my $line, my $aggregated, my $seg_start, my $seg_end, my $text) = split(/\s+/, $_, 6);
  #print "$filename, $seg_start, $seg_end, $text\n";

  if (( $prev_filename ne  $filename ) && ( ";;$prev_filename" ne  $filename)){
    my $_filename = $filename;
    $_filename =~ s/^;;//g;
    next if  not exists $segments{$_filename};
    #print $filename, "\n";
    $prev_filename = $_filename;
    @timestamps = @{$segments{$_filename}};
    #print Dumper(\@timestamps);
    $i=0;
    print RECO "$_filename $_filename 1\n";
  }

  my $max_i=@timestamps;
  while ( ($i < $max_i ) && ($seg_start > @{$timestamps[$i]}[0] ) ) {
    $i+= 1;
  }

  if (($i >= $max_i ) && ($timestamps[$i-1][1]) <= $seg_start ){
    #We are over the start of the last segment -> we assing the last speaker ID
    if ($warn_count < $warn_max) {
      print STDERR "Warning: $prev_filename: the segment from the STM file starts after the last segment from the segments file ends\n";
      print STDERR "Warning: Additional info: STM: ($seg_start, $seg_end), segments file: ($timestamps[$i-1][0] $timestamps[$i-1][1])\n";
      $warn_count += 1;

      if ($warn_count >= $warn_max) {
        print STDERR "Warning: Maximum number of warning reached, not warning anymore...\n"
      }
    }
    #print "$i, $filename, $timestamps[$max_i - 1][2]\n";
    print STMOUT "$filename $line $timestamps[$max_i - 1][2] $seg_start $seg_end $text\n";
  } elsif ( $i == 0  ) {
    if ($warn_count < $warn_max) {
      print STDERR "Warning: $prev_filename: The segment from the STM file start before the first segment from the segments file\n";
      print STDERR "Warning: Additional info: STM: ($seg_start, $seg_end), segments file: ($timestamps[$i][0] $timestamps[$i][1])\n";
      $warn_count += 1;

      if ($warn_count >= $warn_max) {
        print STDERR "Warning: Maximum number of warning reached, not warning anymore...\n"
      }
    }
    #Even the first segment's start time was higher then the stm segment start time
    #That means we do not really know which speaker the stm segment belongs
    print STMOUT "$filename $line $timestamps[$i][2] $seg_start $seg_end $text\n";
    #print "$i, $filename, $timestamps[$i][2]\n";
  } else {
    print STMOUT "$filename $line $timestamps[$i-1][2] $seg_start $seg_end $text\n";
    #print "$i, $filename, $timestamps[$i-1][2]\n";
  }
}

close(STMOUT);
close(STM);
close(RECO);
