#!/usr/bin/env perl
# Copyright 2013  Hong Kong University of Science and Technology (Author: Ricky Chan Ho Yin) 
#                 
# Apache 2.0.
#
# A script to convert Kaldi Chinese words transcription to Chinese characters transcription.
# This is helpful for Chinese character error rate scoring. 
# 
# If no option is applied, by default the script converts the Chinese words transcription to Chinese characters transcription \
# by assuming the input Chinese words/characters are 3 bytes UTF8 code.
# Continuous English/ASCII characters without space are treated as single token.
# 
# When --useword2charmap option is applied, an input Chinese words to Chinese characters mapping table \
# (e.g. a word2char_map likes "195k_chinese_word2char_map") is used for converting the corresponding Chinese words \
# to seperate Chinese characters.
#
# When --encodeoutput option is applied, the script runs like default mode w/o applying option except the  \
# output Chinese characters are in readable encoded format. The output Chinese characters are encoded in a way \
# the same as the opensource HTK toolkit from the Cambridge University Engineering Department.

use POSIX();

sub printUsage {
  print "usage: perl hkust_word2ch_tran.pl [--useword2charmap chinese_word2char_map|--encodeoutput] tran_file \n";
  print "e.g. perl hkust_word2ch_tran.pl tran_file \n";
  print "e.g. perl hkust_word2ch_tran.pl --useword2charmap 195k_chinese_word2char_map tran_file \n";
  print "e.g. perl hkust_word2ch_tran.pl --encodeoutput tran_file \n";
  exit;
} 

sub encodeByteCharacter {
  $enbc = "\\";
  $uchar = ord($_[0]);
  $encrypt1 = (($uchar>>6)&7)+'0';
  $encrypt2 = (($uchar>>3)&7)+'0';
  $encrypt3 = ($uchar&7)+'0';
  $enbc = $enbc."$encrypt1"."$encrypt2"."$encrypt3";
  return $enbc;
}

if(@ARGV < 1 || @ARGV > 3 ) {
  printUsage();
}

$useMapping=0;
$useEncodeoutput=0;

if(@ARGV == 2) {
  if($ARGV[0] ne "--encodeoutput") {
    printUsage();
  }
  $useEncodeoutput=1;
  $tranfile=$ARGV[1];
}
elsif(@ARGV == 3) {
  if($ARGV[0] ne "--useword2charmap") {
    printUsage();
  }
  $useMapping=1;
  $word2charfile=$ARGV[1];
  $tranfile=$ARGV[2];
}
else {
  $tranfile=$ARGV[0];
}

# if Chinese word to character map is provided, read it
if($useMapping) {
  %word2charlist=();
  open(INFILE, $word2charfile) || die("Can't open Chinese word to char map: ".$word2charfile."\n");
  while(<INFILE>){
    chomp;
    @line=split(/\s+/);
    $a=$line[0];
    $b="";
    for($i=1; $i<scalar(@line); $i++) {
      $b=$b . " " . $line[$i];
    }
    $word2charlist{$a}=$b;
  }
  close(INFILE);
}

# process kaldi transcription
open(INFILE, $tranfile) || die("Can't open transcription file ".$tranfile."\n");
while(<INFILE>) {
  chomp;
  @line = split(/\s+/);

  ## utt_id
  print $line[0];

  ## utt_character_word
  for($i=1; $i<scalar(@line); $i++) {
    if($useMapping) {
      if(!exists($word2charlist{$line[$i]})) {
        print " ".$line[$i];
      }
      else {
        print $word2charlist{$line[$i]};
      }
    }
    else {
      @carray = split(//, $line[$i]);
      $wspace=0;
      $l=0;
      while($l<@carray) {
        $c = $carray[$l];
        if(POSIX::isprint($c)) {
          if($wspace) {
            print $c;
          }
          else {
            print " ".$c;
            $wspace=1;
          }
          $l=$l+1;
        }
        else { ## here we find chinese character
          if(!$useEncodeoutput) {
            ## print utf8 chinese character, which should be 3 bytes
            print " ".$carray[$l].$carray[$l+1].$carray[$l+2];
          }
          else {
            ## print 3 bytes utf8 chinese character in readable encoded format
            $enbc1 = encodeByteCharacter($carray[$l]);
            $enbc2 = encodeByteCharacter($carray[$l+1]);
            $enbc3 = encodeByteCharacter($carray[$l+2]);
            print " ".$enbc1.$enbc2.$enbc3;
          }
          $l=$l+3;
          $wspace=0;            
        }
      }
    }
  }
  print "\n";
}
close(INFILE);

