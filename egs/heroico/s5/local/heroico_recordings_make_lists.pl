#!/usr/bin/perl -w

# Copyright 2017 John Morgan
# Apache 2.0.

# heroico_recordings_make_lists.pl - copy make acoustic model training lists

use strict;
use warnings;
use Carp;

use File::Spec;
use File::Copy;
use File::Basename;

BEGIN {
    @ARGV == 1 or croak "USAGE: $0 <PROMPTLIST>";
}

my ($p) = @ARGV;

my $tmpdir = "data/local/tmp/heroico";

system "mkdir -p $tmpdir/recordings";

# input wav file list
my $w = "$tmpdir/wav_list.txt";

# output temporary wav.scp files
my $o = "$tmpdir/recordings/wav.scp";

# output temporary utt2spk files
my $u = "$tmpdir/recordings/utt2spk";

# output temporary text files
my $t = "$tmpdir/recordings/text";

# initialize hash for prompts
my %p = ();

open my $P, '<', $p or croak "problem with $p $!";

# store prompts in hash
LINEA: while ( my $line = <$P> ) {
    chomp $line;
    my ($s,$sent) = split /\t/, $line, 2;
    $p{$s} = $sent;
}
close $P;

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
     my $r = basename $line, ".wav";
     my $s = $dirs[-1];
     my $rid = $s . '_r' . '_' . $r;
     if ( exists $p{$r} ) {
	 print $T "$rid\t$p{$r}\n";
     } elsif ( defined $rid ) {
	 warn  "problem\t$rid";
	 next LINE;
     } else {
	 croak "$line";
     }

     print $O "$rid /$s/$rid sox -r 22050 -e signed -b 16 $line -r 16000 -t wav - |\n";
	print $U "$rid\t${s}_r\n";
}
close $T;
close $O;
close $U;
close $W;
