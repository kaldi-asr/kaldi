#!/usr/bin/env perl
use warnings;

# Copyright  2015 Tokyo Institute of Technology (Authors: Takafumi Moriya and Takahiro Shinozaki)
#            2015 Mitsubishi Electric Research Laboratories (Author: Shinji Watanabe)
# Apache 2.0
# Acknowledgement  This work was supported by JSPS KAKENHI Grant Number 26280055.

# This script is for making word list with morpheme and segment information.

use utf8;
use open IN => ":encoding(utf8)";
use open OUT => ":utf8";
use open ":std";

if (@ARGV != 3){
    die "$0 id.sdb 4lex 4trans\n";
}

$sdb = $ARGV[0];
$lex = $ARGV[1];
$trn = $ARGV[2];
$i = 0;
$p = 0;

open(IN, "nkf -w8 -d $sdb |") || die "$! : $sdb\n";
open(OUTLEX,">$lex") || die "$! : $lex";
open(OUTTRANS,">$trn") || die "$! : $trn";

while (<IN>) {

    chomp;
    @line = split(/\t/, $_);
    $time = $line[3]; # Time information for segment
    $word = $line[5]; # Word
    $num = $line[9]; # Number and point
    ## About morpheme
    $pos = $line[11]; # Part Of Speech
    $acf = $line[12]; # A Conjugated Form
    $kacf = $line[13]; # Kind of A Conjugated Form
    $kav = $line[14]; # Kind of Aulxiliary Verb
    $ec = $line[15]; # Euphonic Change
    $other = $line[16]; # Other information
    $pron = $line[10]; # Pronunciation for lexicon

    foreach $var($pos,$acf,$kacf,$kav,$ec,$other){
    $var = &checkempty($var);
    }    

    $morph ="$pos/$acf/$kacf/$kav/$ec"; # Exclude "other" information
#    $morph ="$pos/$acf/$kacf/$kav/$ec/$other"; # Include "other" information

    ## Arrange word format
    if(($word =~ /A/) || ($i != 0)){
	if((($word =~ /[\ゼロ,\０,\零,\一,\二,\三,\四,\五,\六,\七,\八,\九,\十,\百,\千,\．]/)&&($word =~ /A/)) || ($p == 1)){
	    $p = 1;
	    if(($word =~ /;/) && ($i == 0)){
		$word = $num;
            }
	    elsif(($word =~ /\;/)&&($i != 0)){
		$word = $num;
		$i = 0;
		$p = 0;
	    }
	    else{
		$word = $num;
		$i++;
	    }
	}
	else{
	    if(($word =~ /;/) && ($i == 0)){
		@word_type = split(/\;/,$word);
		$word = $word_type[1];
		$word =~ tr/[\x00-\x7F]//d;
	    }
	    elsif(($word =~ /\;/)&&($i != 0)){
		@word_type = split(/;/,$word);
		$word = $word_type[1];
		$word =~ tr/[\x00-\x7F]//d;
		$opron = $pron ;
		$pron = "";
		for($i=0; $i<$#epron+1; $i++){
		    $pron .= "$epron[$i]";
		}
		$pron .= $opron;
		$i = 0;
	    }
	    else{
		$epron[$i] = $pron;
		$pron = "";
		$i++;
		$word = skip_word;
	    }
	}
    }
    else{
	$word =~ tr/[\x00-\x7F]//d;
    }


    ### For extracting pronunciation operation. You select case either 1 or 2. ###
    $pron_case=1;
    ## Case 1 (**Default and better select.)
    # e.g.
    # Input  : クールエディット+(W (? クーレリットル);クールエディット)+名詞
    # ->
    # Output : クールエディット+クールエディット+名詞

    ## Case 2 (**If you use this case, it probably can't make HCLG.fst well because of so many different pronunciation.)
    # e.g.
    # Input  : クールエディット+(W (? クーレリットル);クールエディット)+名詞
    # ->
    # Output : クールエディット+クーレリットル+名詞

    if($pron_case == 1){
	## Case 1 processing
	if ($pron =~ /[\x00-\x7f]/){
	    if (($pron =~ /^\(/)&&($pron =~ /\;/)) {
		@lipron=split(/\;/, $pron);
		$pron=$lipron[$#lipron];
		$pron =~ tr/[\x00-\x7F]//d;
	    }
	    elsif ($pron =~ /^(\S)+\(/){
		@lipre1=split(/\(/,$pron);
		$lpw11 = $lipre1[0];
		$lpw12 = $lipre1[1];
		if ($lpw12 =~ /\;/) {
		    @liprefin1=split(/\;/,$lpw12);
		    $liprefin12=$liprefin1[$#liprefin1];
		    $liprefin12 =~ tr/[\x00-\x7F]//d;
		    $pron = "$lpw11$liprefin12";
		}
		else{
		    $pron =~ tr/[\x00-\x7F]//d;
		}
	    }
	    elsif ($pron =~ /\)(\S)+$/){
		@lipre2=split(/\)/,$pron);
		$lpw21 = $lipre2[0];
		$lpw22 = $lipre2[1];
		if ($lpw21 =~ /\;/) {
		    @liprefin2=split(/\;/,$lpw21);
		    $liprefin22=$liprefin2[$#liprefin2];
		    $liprefin22 =~ tr/[\x00-\x7F]//d;
		    $pron = "$liprefin22$lpw22";
                }
		else{
		    $pron =~ tr/[\x00-\x7F]//d;
		}
	    }
	    else{
		$pron =~ tr/[\x00-\x7F]//d;
	    }
	}
    }
    else{
	## Case 2 processing
	if ($pron =~ /[\x00-\x7f]/){
	    if (($pron =~ /^\(/)&&($pron =~ /\)$/)) {
		@lipron=split(/\;/, $pron);
		$pron=$lipron[0];
		$pron =~ tr/[\x00-\x7F]//d;
	    }
	    elsif ($pron =~ /^(\S)+\(/){
		@lipre1=split(/\(/,$pron);
		$lpw11 = $lipre1[0];
		$lpw12 = $lipre1[1];
		if ($lpw12 =~ /\;/) {
		    @liprefin1=split(/\;/,$lpw12);
		    $liprefin12=$liprefin1[0];
		    $liprefin12 =~ tr/[\x00-\x7F]//d;
		    $pron = "$lpw11$liprefin12";
		}
		else{
		    $pron =~ tr/[\x00-\x7F]//d;
		}
	    }
	    elsif ($pron =~ /\)(\S)+$/){
		@lipre2=split(/\)/,$pron);
		$lpw21 = $lipre2[0];
		$lpw22 = $lipre2[1];
		if ($lpw21 =~ /\;/) {
		    @liprefin2=split(/\;/,$lpw21);
		    $liprefin22=$liprefin2[0];
		    $liprefin22 =~ tr/[\x00-\x7F]//d;
		    $pron = "$liprefin22$lpw22";
		}
		else{
		    $pron =~ tr/[\x00-\x7F]//d;
		}
	    }
	    else{
		$pron =~ tr/[\x00-\x7F]//d;
	    }
	}
    }

    ## Arrangement and Normarization part
    # Remove unnecessary tag and fix absurd pronuciation
    $pron =~ tr/[\x00-\x7F]//d;
    $pron =~ tr/\笑//d;
    $pron =~ tr/\息//d;
    $pron =~ tr/\咳//d;
    $pron =~ tr/\泣//d;
    $pron =~ tr/\×//d;
    $pron =~ s/\雑\音//g;
    $pron =~ s/\ン\ー/\ン/g;
    $pron =~ s/\ン\ー/\ン/g;

    # Modify minor bug words.
    $word =~ s/\・$//g;
    $word =~ s/\ん\ー/\ん/g;
    $word =~ s/\ン\ー/\ン/g;
    $word =~ s/\ん\ー/\ん/g; # In case of "んーー"
    $word =~ s/\ン\ー/\ン/g; #

    # Arrange and normarize option for morpheme and word:
    # By using this option, you may obtain better result than without it.
    # If you want to use it, change $morph_opt and $word_opt "" to "1" respectively.
    $morph_opt = 1;
    $word_opt = 0;
    if($morph_opt == 1){
	#$morph =~ s/\/形容詞型//;
	#$morph =~ s/\/文語形容詞型//;
	#$morph =~ s/\/文語/\//;
	#$morph =~ s/\/\//\/文語\//;
	#$morph =~ s/\/$//;
	#$morph =~ s/\/語幹//;
	#$morph =~ s/\/省略//;
	$morph =~ s/\Ａ//;
	$morph =~ s/１//g;
	$morph =~ s/２//g;
	$morph =~ s/３//g;
	$morph =~ s/４//g;
    }

    if($word_opt == 1){
        $word =~ s/^\ゼロ$/\０/g;
        $word =~ s/^\零$/\０/g;
    }


    $morph =~ s/\/\//\//g;
    $morph =~ s/\/\//\//g;
    $morph =~ s/\/\//\//g;
    $morph =~ s/\/\//\//g;
    $morph =~ s/\/$//g;
    # Replace Zenkaku-Space to Zenkaku-Underscore
    # Input: っしゃっ+動詞/ラ行五段/連用形/促音便　省略 r a q sh a q
    # ->
    # Output: っしゃっ+動詞/ラ行五段/連用形/促音便＿省略 r a q sh a q
    $morph =~ s/　/＿/g;

    unless( $word =~ /skip_word/){
	if ($word && $pos){
	    print OUTLEX  "$word+$pron+$morph\n";
	    print OUTTRANS "$time $word+$morph\n";
	}
    }
}

sub checkempty{
    my($judge) = @_;
    if(!$judge){
	return '';
    }
    return $judge;
}
