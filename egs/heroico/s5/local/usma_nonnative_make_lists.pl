#!/usr/bin/env perl

# Copyright 2017 John Morgan
# Apache 2.0.

#usma_nonnative_make_lists.pl - make acoustic model training lists

use strict;
use warnings;
use Carp;

use File::Spec;
use File::Copy;
use File::Basename;

my $tmpdir = "data/local/tmp/usma/nonnative";

# input wav file list
my $w = "$tmpdir/wav_list.txt";

# output temporary wav.scp files
my $o = "$tmpdir/wav.scp";

# output temporary utt2spk files
my $u = "$tmpdir/utt2spk";

# output temporary text files
my $t = "$tmpdir/text";

# initialize hash for prompts
my %p = ();

# store prompts in hash
LINEA: while ( my $line = <> ) {
    chomp $line;
    my ($num,$sent) = split /\t/, $line, 2;
    $p{$num} = $sent;
}

open my $W, '<', $w or croak "problem with $w $!";
open my $O, '+>', $o or croak "problem with $o $!";
open my $U, '+>', $u or croak "problem with $u $!";
open my $T, '+>', $t or croak "problem with $t $!";

 LINE: while ( my $line = <$W> ) {
     chomp $line;
     next LINE unless ( $line =~ /nonnative/ );
     my ($volume,$directories,$file) = File::Spec->splitpath( $line );
     my @dirs = split /\//, $directories;
     my $r = basename $line, ".wav";
     next LINE unless ( $r =~ /^s/ );
     my $s = $dirs[-1];
     my ($nativeness,$gender,$country,$weight,$age,$height,$dlpt,$idx) = split /\-/, $s, 9;
     $s = $nativeness . '_' . $gender . '_' . $country . '_' . $weight . '_' . $age . '_' . $height . '_' . $dlpt . '_' . $idx;
     my $rid = $s . '_' . $r;
     if ( exists $p{$r} ) {
	 print $T "$rid $p{$r}\n";
     } elsif ( defined $rid ) {
	 warn  "problem\t$rid";
	 next LINE;
     } else {
	 croak "$line";
     }

     print $O "$rid sox -r 22050 -e signed -b 16 $line -r 16000 -t wav - |\n";
     print $U "$rid $s\n";
}
close $T;
close $O;
close $U;
close $W;
