#!/usr/bin/env perl

###########################################################################################
# This script was copied from egs/ami/s5/local/ami_split_segments.pl
# The source commit was e69198c3dc5633f98eb88e1cdf20b2521a598f21
# No changes were made
###########################################################################################

# Copyright 2014  University of Edinburgh (Author: Pawel Swietojanski)

# The script - based on punctuation times - splits segments longer than #words (input parameter)
# and produces bit more more normalised form of transcripts, as follows
# MeetID Channel Spkr stime etime transcripts 

#use List::MoreUtils 'indexes';
use strict;
use warnings;

sub split_transcripts;
sub normalise_transcripts;

sub merge_hashes {
   my ($h1, $h2) = @_;
   my %hash1 = %$h1; my %hash2 = %$h2;
   foreach my $key2 ( keys %hash2 ) {
      if( exists $hash1{$key2} ) {
         warn "Key [$key2] is in both hashes!";
         next;
      } else {
        $hash1{$key2} = $hash2{$key2};
      }
   }
   return %hash1;
}

sub print_hash {
   my ($h) = @_;
   my %hash = %$h;
   foreach my $k (sort keys %hash) {
      print "$k : $hash{$k}\n";
   }
}

sub get_name {
  #no warnings;
  my $sname = sprintf("%07d_%07d", $_[0]*100, $_[1]*100) || die 'Input undefined!';
  #use warnings;
  return $sname;
}

sub split_on_comma {

   my ($text, $comma_times, $btime, $etime, $max_words_per_seg)= @_;
   my %comma_hash = %$comma_times;

   print "Btime, Etime : $btime, $etime\n";
  
   my $stime = ($etime+$btime)/2; #split time
   my $skey = "";
   my $otime = $btime;
   foreach my $k (sort {$comma_hash{$a} cmp $comma_hash{$b} } keys %comma_hash) {
      print "Key : $k : $comma_hash{$k}\n";
      my $ktime = $comma_hash{$k};
      if ($ktime==$btime) { next; }
      if ($ktime==$etime) { last; }
      if (abs($stime-$ktime)/2<abs($stime-$otime)/2) {
         $otime = $ktime;
         $skey = $k;
      }
   }
  
   my %transcripts = ();

   if (!($skey =~ /[\,][0-9]+/)) {
      print "Cannot split into less than $max_words_per_seg words! Leaving : $text\n";
      $transcripts{get_name($btime, $etime)}=$text;
      return %transcripts;
   }   

   print "Splitting $text on $skey at time $otime (stime is $stime)\n";  
   my @utts1 = split(/$skey\s+/, $text);
   for (my $i=0; $i<=$#utts1; $i++) {
     my $st = $btime;
     my $et = $comma_hash{$skey};
     if ($i>0) { 
        $st=$comma_hash{$skey}; 
        $et = $etime; 
     }
     my (@utts) = split (' ', $utts1[$i]);
     if ($#utts < $max_words_per_seg) {
        my $nm = get_name($st, $et);
        print "SplittedOnComma[$i]: $nm : $utts1[$i]\n";
        $transcripts{$nm} = $utts1[$i];
     } else {
        print 'Continue splitting!';
        my %transcripts2 = split_on_comma($utts1[$i], \%comma_hash, $st, $et, $max_words_per_seg);
        %transcripts = merge_hashes(\%transcripts, \%transcripts2);
     }
   }
   return %transcripts;
}
  
sub split_transcripts {
  @_ == 4 || die 'split_transcripts: transcript btime etime max_word_per_seg';

  my ($text, $btime, $etime, $max_words_per_seg) = @_;
  my (@transcript) = @$text;

  my (@punct_indices) = grep { $transcript[$_] =~ /^[\.,\?\!\:]$/ } 0..$#transcript;
  my (@time_indices) = grep { $transcript[$_] =~ /^[0-9]+\.[0-9]*/ } 0..$#transcript;
  my (@puncts_times) = delete @transcript[@time_indices]; 
  my (@puncts) = @transcript[@punct_indices];

  if ($#puncts_times != $#puncts) {
     print 'Ooops, different number of punctuation signs and timestamps! Skipping.';
     return ();
  }
 
  #first split on full stops
  my (@full_stop_indices) = grep { $puncts[$_] =~ /[\.\?]/ } 0..$#puncts;
  my (@full_stop_times) = @puncts_times[@full_stop_indices];

  unshift (@full_stop_times, $btime);
  push (@full_stop_times, $etime);

  my %comma_puncts = ();
  for (my $i=0, my $j=0;$i<=$#punct_indices; $i++) {
     my $lbl = "$transcript[$punct_indices[$i]]$j";
     if ($lbl =~ /[\.\?].+/) { next; }
     $transcript[$punct_indices[$i]] = $lbl;
     $comma_puncts{$lbl} = $puncts_times[$i];
     $j++;
  }

  #print_hash(\%comma_puncts);
  
  print "InpTrans : @transcript\n";
  print "Full stops: @full_stop_times\n";

  my @utts1 = split (/[\.\?]/, uc join(' ', @transcript));
  my %transcripts = ();
  for (my $i=0; $i<=$#utts1; $i++) {
     my (@utts) = split (' ', $utts1[$i]);
     if ($#utts < $max_words_per_seg) {
        print "ReadyTrans: $utts1[$i]\n";
        $transcripts{get_name($full_stop_times[$i], $full_stop_times[$i+1])} = $utts1[$i];
     } else {
        print "TransToSplit: $utts1[$i]\n";
        my %transcripts2 = split_on_comma($utts1[$i], \%comma_puncts, $full_stop_times[$i], $full_stop_times[$i+1], $max_words_per_seg);
        print "Hash TR2:\n"; print_hash(\%transcripts2);
        print "Hash TR:\n"; print_hash(\%transcripts);
        %transcripts = merge_hashes(\%transcripts, \%transcripts2);
        print "Hash TR_NEW : \n"; print_hash(\%transcripts);
     }
  }   
  return %transcripts;
}

sub normalise_transcripts {
   my $text = $_[0];

   #DO SOME ROUGH AND OBVIOUS PRELIMINARY NORMALISATION, AS FOLLOWS
   #remove the remaining punctation labels e.g. some text ,0 some text ,1
   $text =~ s/[\.\,\?\!\:][0-9]+//g;
   #there are some extra spurious puncations without spaces, e.g. UM,I, replace with space
   $text =~ s/[A-Z']+,[A-Z']+/ /g;
   #split words combination, ie. ANTI-TRUST to ANTI TRUST (None of them appears in cmudict anyway)
   #$text =~ s/(.*)([A-Z])\s+(\-)(.*)/$1$2$3$4/g;
   $text =~ s/\-/ /g;
   #substitute X_M_L with X. M. L. etc.
   $text =~ s/\_/. /g;
   #normalise and trim spaces
   $text =~ s/^\s*//g;
   $text =~ s/\s*$//g;
   $text =~ s/\s+/ /g;
   #some transcripts are empty with -, nullify (and ignore) them
   $text =~ s/^\-$//g;
   $text =~ s/\s+\-$//;
   # apply few exception for dashed phrases, Mm-Hmm, Uh-Huh, etc. those are frequent in AMI
   # and will be added to dictionary
   $text =~ s/MM HMM/MM\-HMM/g;
   $text =~ s/UH HUH/UH\-HUH/g;

   return $text;
}

if (@ARGV != 2) {
  print STDERR "Usage: ami_split_segments.pl <meet-file> <out-file>\n";
  exit(1);
}

my $meet_file = shift @ARGV;
my $out_file = shift @ARGV; 
my %transcripts = ();

open(W, ">$out_file") || die "opening output file $out_file";
open(S, "<$meet_file") || die "opening meeting file $meet_file";

while(<S>) {

  my @A = split(" ", $_);
  if (@A < 9) { print "Skipping line @A"; next; }
  
  my ($meet_id, $channel, $spk, $channel2, $trans_btime, $trans_etime, $aut_btime, $aut_etime) = @A[0..7];
  my @transcript = @A[8..$#A];
  my %transcript = split_transcripts(\@transcript, $aut_btime, $aut_etime, 30); 

  for my $key (keys %transcript) {
    my $value = $transcript{$key};
    my $segment = normalise_transcripts($value); 
    my @times = split(/\_/, $key);
    if ($times[0] >= $times[1]) {
       print "Warning, $meet_id, $spk, $times[0] > $times[1]. Skipping. \n"; next;
    }
    if (length($segment)>0) {
       print W join " ", $meet_id, "H0${channel2}", $spk, $times[0]/100.0, $times[1]/100.0, $segment, "\n";
    }
  }

}
close(S);
close(W);

print STDERR "Finished."
