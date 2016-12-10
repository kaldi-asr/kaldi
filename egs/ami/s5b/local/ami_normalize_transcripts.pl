#!/usr/bin/env perl

# Copyright 2014  University of Edinburgh (Author: Pawel Swietojanski)
#           2016  Vimal Manohar

# The script - based on punctuation times - splits segments longer than #words (input parameter)
# and produces bit more more normalised form of transcripts, as follows
# MeetID Channel Spkr stime etime transcripts 

#use List::MoreUtils 'indexes';
use strict;
use warnings;

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
  
sub normalise_transcripts {
   my $text = $_;

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

while(<>) {
  chomp;
  print normalise_transcripts($_) . "\n";
}

