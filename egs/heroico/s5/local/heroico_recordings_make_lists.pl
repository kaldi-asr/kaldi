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
my $w = "$tmpdir/wav_list.txt";

# output temporary wav.scp files
my $o_train = "$tmpdir/recordings/train/wav.scp";
my $o_test = "$tmpdir/recordings/devtest/wav.scp";

# output temporary utt2spk files
my $u_train = "$tmpdir/recordings/train/utt2spk";
my $u_test = "$tmpdir/recordings/devtest/utt2spk";

# output temporary text files
my $t_train = "$tmpdir/recordings/train/text";
my $t_test = "$tmpdir/recordings/devtest/text";

# initialize hash for prompts
my %p = ();

# store prompts in hash
LINEA: while ( my $line = <> ) {
    chomp $line;
    my ($s,$sent) = split /\t/, $line, 2;
    $p{$s} = $sent;
}

open my $W, '<', $w or croak "problem with $w $!";
open my $OT, '+>', $o_train or croak "problem with $o_train $!";
open my $OE, '+>', $o_test or croak "problem with $o_test $!";
open my $UT, '+>', $u_train or croak "problem with $u_train $!";
open my $UE, '+>', $u_test or croak "problem with $u_test $!";
open my $TT, '+>', $t_train or croak "problem with $t_train $!";
open my $TE, '+>', $t_test or croak "problem with $t_test $!";

 LINE: while ( my $line = <$W> ) {
     chomp $line;
     next LINE if ($line =~ /Answers/ );
     next LINE unless ( $line =~ /Recordings/ );
     my ($volume,$directories,$file) = File::Spec->splitpath( $line );
     my @dirs = split /\//, $directories;
     my $r = basename $line, ".wav";
     my $s = $dirs[-1];
     my $rid = $s . '_r' . '_' . $r;
     if ( ( $r >= 355 ) and ( $r < 561 ) ) {
	 if ( exists $p{$r} ) {
	     print $TE "$rid $p{$r}\n";
	 } elsif ( defined $rid ) {
	     warn  "problem\t$rid";
	     next LINE;
	 } else {
	     croak "$line";
	 }
	 print $OE "$rid sox -r 22050 -e signed -b 16 $line -r 16000 -t wav - |\n";
	 print $UE "$rid ${s}_r\n";
     } elsif ( ( $r < 355 ) or ( $r > 560 ) ) {
	 if ( exists $p{$r} ) {
	     print $TT "$rid $p{$r}\n";
	 } elsif ( defined $rid ) {
	     warn  "problem\t$rid";
	     next LINE;
	 } else {
	     croak "$line";
	 }
	 print $OT "$rid sox -r 22050 -e signed -b 16 $line -r 16000 -t wav - |\n";
	 print $UT "$rid ${s}_r\n";
     }
}
close $TT;
close $OT;
close $UT;
close $TE;
close $OE;
close $UE;
close $W;
