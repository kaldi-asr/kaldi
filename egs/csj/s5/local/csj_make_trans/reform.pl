#! /usr/bin/perl -w                                                           

# Copyright  2015 Tokyo Institute of Technology (Authors: Takafumi Moriya and Takahiro Shinozaki)
#            2015 Mitsubishi Electric Research Laboratories (Author: Shinji Watanabe)
# Apache 2.0
# Acknowledgement  This work was supported by JSPS KAKENHI Grant Number 26280055.

# This script is to make lexicon for KALDI format.

while (<>){
    chomp;
    @line=split(/\t/, $_);
   
    $word_m=$line[0]; # Word + Morpheme
    $phone=$line[$#line]; # Phoneme
    
    chomp;
    @list=split(/\+/, $word_m);
    $word =$list[0];
    $word =~ s/\ん\ー/\ん/g;
    $word =~ s/\ヮ/\ワ/g;
    $word =~ s/\ゎ/\わ/g;
    $morph=$list[$#list];
    $word_m="$word+$morph";
    
    # Fix and remove obstructive tag and component
    $phone =~ s/\×(.)+//g;
    $phone =~ s/\t\:/\t/g;
    $phone =~ s/\<sp\>//g;
    $phone =~ s/息//g;
    $phone =~ s/笑//g;
    $phone =~ s/咳//g;
    $phone =~ s/N:/N/g;
    $phone =~ s/\ヮ/w a/g;
    $phone =~ s/^: //g;
    $phone =~ s/ : //g;
    $phone =~ s/\[(.)+\]//g;
    $phone =~ s/^q $//g;    
    $word_m =~ s/\×(.)+//g;

    if($word_m && $phone){
	print "$word_m $phone\n";
    }
    else{
	print "";
    }
}
