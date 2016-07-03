#!/usr/bin/env perl
use warnings;

# Copyright  2015 Tokyo Institute of Technology (Authors: Takafumi Moriya and Takahiro Shinozaki)
#            2015 Mitsubishi Electric Research Laboratories (Author: Shinji Watanabe)
# Apache 2.0
# Acknowledgement  This work was supported by JSPS KAKENHI Grant Number 26280055.

# This script converts Katakana phonetic alphabet to phones.

use utf8;
use open IO => ":utf8";
use open ":std";

use Getopt::Std;
getopt('o:e:p:');

if (@ARGV != 1 ) {
    die "Usage: vocab2dic.pl [-o OUTFILE (default htkprondic)] [-p PHONELIST (default kana2phone)] VOCABFILE\n";
}

if (!$opt_p) {
    $0 =~ /^(.*\/)[^\/]*$/;
    $script_dir = $1;

    $opt_p = $script_dir . "kana2phone";
}

if (!$opt_o) {
    $opt_o = "htkprondic";
}

if (!$opt_e) {
    $opt_e = "ERROR";
}

open(DIC, "> $opt_o") || die "can't write $opt_o\n";
open(ER, "> $opt_e") || die "can't write $opt_e\n";

open(PHONES, $opt_p);
while(<PHONES>) {
    chomp;

    ($kana, $phone) = split('\+', $_);
    $kana2phone{$kana} = $phone;
}

# main function
$line_num = 0;
while (<>) {
    next if /^#/;
    chomp;
    $line_num++;

    if (/^<s>/) {
	print DIC "<s>\t[]\tsilB\n";
	next;
    }
    if (/^<\/s>/) {
	print DIC "</s>\t[]\tsilE\n";
	next;
    }
    if (/^<sp>/) {
	print DIC "<sp>\t[]\tsp\n";
	next;
    }

    $word = $_;
    ($morph, $kana) = split('\+', $word);

    ($phoneseq, $flg) = &kanaseq2phoneseq($kana);

    print DIC "$word\t[$morph]\t$phoneseq\n";
    print ER "$line_num: $word [$morph] $phoneseq\n" if $flg;
}

# sub functions

sub usage() {
    print "perl vocab2dic.pl <OPTS> file.vocab\n";
    print "  OPTS : [-h] [-o fname] [-e fname] [-p fname]\n";
    print "    -h       : show usage\n";
    print "    -o fname : write a HTK format dictionary to a file \"fnane\"\n";
    print "                 default \"fname\" is \"htkprondic\"\n";
    print "    -e fname : write invalid entries to a file \"fname\"\n";
    print "                 default \"fname\" is \"ERROR\"\n";
    print "    -p fname : specify a kana-to-phone-transformation file\n";
    print "                 default \"fname\" is \"kana2phone\"\n";
    exit(-1);
}

sub kanaseq2phoneseq($) {
    my($kanaseq) = @_;
    my($flg, $phoneseq, $syllable, @chars, @syllables);

    $flg = 0;
    $phoneseq = "";
    $syllable = "";
    @chars = ();
    @syllables = ();

    @chars = split(//, $kanaseq);

    foreach $char (@chars) {
	if ($char =~ /[ァィゥェォャュョ]/) {
	    $syllable .= $char;
	} else {
	    if (!$syllable) {
		$syllable = $char;
	    } else {
		$syllable .= " " . $char;
	    }
	}
    }

    @syllables = split(' ', $syllable);

    foreach $syllable (@syllables) {
	if ($kana2phone{$syllable}) {
	    if ($kana2phone{$syllable} eq ": ") {
		chop($phoneseq);
	    }
	    $phoneseq .= $kana2phone{$syllable};
	} else {
	    $flg = 1;
	    $phoneseq .= $syllable;
	}
    }

    if ($phoneseq =~ /^: $/ || $phoneseq =~ /^q $/) {
	$flg = 1;
    }

    return ($phoneseq, $flg);
}
