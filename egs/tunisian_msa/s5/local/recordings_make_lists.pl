#!/usr/bin/env perl

# Copyright 2018 John Morgan
# Apache 2.0.

# recordings_make_lists.pl - make acoustic model training lists

use strict;
use warnings;
use Carp;

use File::Spec;
use File::Copy;
use File::Basename;

my $tmpdir = "data/local/tmp/tunis";

system "mkdir -p $tmpdir/recordings";

# input wav file list
my $w = "$tmpdir/recordings_wav.txt";

# output temporary wav.scp files
my $o = "$tmpdir/recordings/wav.scp";

# output temporary utt2spk files
my $u = "$tmpdir/recordings/utt2spk";

# output temporary text files
my $t = "$tmpdir/recordings/text";

# initialize hash for prompts
my %p = ();

# store prompts in hash
LINEA: while ( my $line = <> ) {
    chomp $line;
    my ($s,$sent) = split /\t/, $line, 2;
    $p{$s} = $sent;
}

open my $W, '<', $w or croak "problem with $w $!";
open my $O, '+>', $o or croak "problem with $o $!";
open my $U, '+>', $u or croak "problem with $u $!";
open my $T, '+>', $t or croak "problem with $t $!";

 LINE: while ( my $line = <$W> ) {
     chomp $line;
     next LINE if ($line =~ /Answers/ );
     next LINE unless ( $line =~ /Recordings/ );
     my ($volume,$directories,$file) = File::Spec->splitpath( $line );
     my @dirs = split /\//, $directories;
     my $machine = $dirs[-3];
     my $r = basename $line, ".wav";
     my $s = $dirs[-1];
     my $rid = $machine . '_' . $s . '_r_' . $r;
     if ( exists $p{$r} ) {
	 print $T "$rid\t$p{$r}\n";
     } elsif ( defined $rid ) {
	 warn  "problem\t$rid";
	 next LINE;
     } else {
	 croak "$line";
     }

     print $O "$rid sox $line -t wav - |\n";
	print $U "$rid\t${machine}_${s}_r\n";
}
close $T;
close $O;
close $U;
close $W;
