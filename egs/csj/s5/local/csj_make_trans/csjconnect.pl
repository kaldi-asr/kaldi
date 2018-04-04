#!/usr/bin/env perl
use warnings;

# Copyright  2015 Tokyo Institute of Technology (Authors: Takafumi Moriya and Takahiro Shinozaki)
#            2015 Mitsubishi Electric Research Laboratories (Author: Shinji Watanabe)
# Apache 2.0
# Acknowledgement  This work was supported by JSPS KAKENHI Grant Number 26280055.

# Connects CSJ segments to each moderate length utterance units by csj2kaldi4m.pl.

use utf8;
use open IO => ":utf8";
use open ":std";

if (@ARGV != 4) {
    die "$0 gap maxlen file spk_id\n";
}

$gap = $ARGV[0];
$maxlen = $ARGV[1];
$file = $ARGV[2];
$spk = $ARGV[3];

if ($file eq '-') {
    $in = STDIN;
} else {
    open($in, $file) || die "$! : $file\n";
}
$psgid = -1;
$pspk_id = "";
$pend = 0;
$line = "";

while (<$in>) {
    chomp;
    if (! /(\d+) ([\d\.]+)-([\d\.]+) (.):-\d+-\d+ (.+)/) {
	die  "Unexpected format: $_\n";
    }
    $sgid = $1;
    $start = $2;
    $end = $3;
    $pch = $4;
    $wpp = $5;
    if ( $spk =~ /^D/ ){
	$ch = "\-$pch";
    } else {
	$ch = "";
    }
    $spk_id = "$spk$ch";


    if ($psgid == -1) {
	$ostart = $start;
	$osgid = $sgid;
	$ospk_id = $spk_id;
	$line = "$wpp ";
    } elsif ($psgid eq $sgid && $pspk_id eq $spk_id) {
	$line .= "$wpp ";
    } else {
	if ($gap < $start - $pend || $maxlen < $pend - $ostart || $ospk_id ne $spk_id ) {
	    if ($opt_t) {
		print "$osgid $ostart $pend\n";
	    } else {
		unless($line=~ /\×/){
		    print "$ospk_id\_$osgid $ostart $pend <s> $line</s>\n";
		}
	    }
	    $ostart = $start;
	    $osgid = $sgid;
	    $ospk_id = $spk_id;
	    $line = "$wpp ";
	} else {
	    $line .= "<sp> $wpp ";
	}
    }

    $psgid = $sgid;
    $pspk_id = $spk_id;
    $pend = $end;
}

if ($line ne "") {
    if ($opt_t) {
	print "$osgid $ostart $end\n";
    } else {
	unless($line =~ /\×/){
	    print "$ospk_id\_$osgid $ostart $end <s> $line</s>\n";
	}
    }
}
