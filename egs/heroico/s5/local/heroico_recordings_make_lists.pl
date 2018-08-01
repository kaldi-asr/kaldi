#!/usr/bin/env perl

# Copyright 2017 John Morgan
# Apache 2.0.

# heroico_recordings_make_lists.pl - make acoustic model training lists

use strict;
use warnings;
use Carp;

use File::Spec;
use File::Copy;
use File::Basename;

my $tmpdir = "data/local/tmp/heroico";

system "mkdir -p $tmpdir/recordings/train";
system "mkdir -p $tmpdir/recordings/devtest";

# input wav file list
my $input_wav_list = "$tmpdir/wav_list.txt";

# output temporary wav.scp files
my $train_wav_scp = "$tmpdir/recordings/train/wav.scp";
my $test_wav_scp = "$tmpdir/recordings/devtest/wav.scp";

# output temporary utt2spk files
my $train_uttspk = "$tmpdir/recordings/train/utt2spk";
my $test_uttspk = "$tmpdir/recordings/devtest/utt2spk";

# output temporary text files
my $train_text = "$tmpdir/recordings/train/text";
my $test_text = "$tmpdir/recordings/devtest/text";

# initialize hash for prompts
my %prompts = ();

# store prompts in hash
LINEA: while ( my $line = <> ) {
    chomp $line;
    my ($prompt_id,$prompt) = split /\t/, $line, 2;
    # pad the prompt id with zeroes
    my $pid = "";
    if ( $prompt_id < 10 ) {
	$pid = '0000' . $prompt_id;
    } elsif ( $prompt_id < 100 ) {
	$pid = '000' . $prompt_id;
    } elsif ( $prompt_id < 1000 ) {
	$pid = '00' . $prompt_id;
    }
    $prompts{$pid} = $prompt;
}

open my $WVL, '<', $input_wav_list or croak "problem with $input_wav_list $!";
open my $TRNWSCP, '+>', $train_wav_scp or croak "problem with $train_wav_scp $!";
open my $TSTWSCP, '+>', $test_wav_scp or croak "problem with $test_wav_scp $!";
open my $TRNUTTSPK, '+>', $train_uttspk or croak "problem with $train_uttspk $!";
open my $TSTUTTSPK, '+>', $test_uttspk or croak "problem with $test_uttspk $!";
open my $TRNTXT, '+>', $train_text or croak "problem with $train_text $!";
open my $TSTTXT, '+>', $test_text or croak "problem with $test_text $!";

 LINE: while ( my $line = <$WVL> ) {
     chomp $line;
     next LINE if ($line =~ /Answers/ );
     next LINE unless ( $line =~ /Recordings/ );
     my ($volume,$directories,$file) = File::Spec->splitpath( $line );
     my @dirs = split /\//, $directories;
     my $utt_id = basename $line, ".wav";
     # pad the utterance id with zeroes
     my $utt = "";
     if ( $utt_id < 10 ) {
     $utt = '0000' . $utt_id;
} elsif ( $utt_id < 100 ) {
    $utt = '000' . $utt_id;
} elsif ( $utt_id < 1000 ) {
    $utt = '00' . $utt_id;
}
     my $spk_id = $dirs[-1];
     # pad the speaker id with zeroes
     my $spk = "";
     if ( $spk_id < 10 ) {
	 $spk = '000' . $spk_id;
     } elsif ( $spk_id < 100 ) {
	 $spk = '00' . $spk_id;
     } elsif ( $spk_id < 1000 ) {
	 $spk = '0' . $spk_id;
     }
     my $spk_utt_id = $spk . '_' . $utt;
     if ( ( $utt_id >= 355 ) and ( $utt_id < 561 ) ) {
if ( exists $prompts{$utt} ) {
	     print $TSTTXT "$spk_utt_id $prompts{$utt}\n";
	 } elsif ( defined $spk_utt_id ) {
	     warn  "problem\t$spk_utt_id";
	     next LINE;
	 } else {
	     croak "$line";
	 }
	 print $TSTWSCP "$spk_utt_id sox -r 22050 -e signed -b 16 $line -r 16000 -t wav - |\n";
	 print $TSTUTTSPK "$spk_utt_id $spk\n";
     } elsif ( ( $utt_id < 355 ) or ( $utt_id > 560 ) ) {
	 if ( exists $prompts{$utt} ) {
	     print $TRNTXT "$spk_utt_id $prompts{$utt}\n";
	 } elsif ( defined $spk_utt_id ) {
	     warn  "problem\t$spk_utt_id";
	     next LINE;
	 } else {
	     croak "$line";
	 }
	 print $TRNWSCP "$spk_utt_id sox -r 22050 -e signed -b 16 $line -r 16000 -t wav - |\n";
	 print $TRNUTTSPK "$spk_utt_id $spk\n";
     } 
}
close $TRNTXT;
close $TRNWSCP;
close $TRNUTTSPK;
close $TSTTXT;
close $TSTWSCP;
close $TSTUTTSPK;
close $WVL;
