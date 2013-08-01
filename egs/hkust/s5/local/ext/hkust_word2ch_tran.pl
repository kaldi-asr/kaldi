#!/usr/bin/perl
# Copyright Hong Kong University of Science and Technology (Author: Ricky Chan) 2013.
#
# A script to convert Kaldi Chinese word transcription to Chinese character transcription using a word2char mapping (e.g. a word2char_map likes "195k_chinese_word2char_map")
# This is helpful for Chinese character word error rate scoring 

if($#ARGV+1 != 2) {
  print "usage: perl hkust_word2char_tran.pl chinese_word2char_map tran_file \n";
  exit;
}

$word2charfile=$ARGV[0];
$tranfile=$ARGV[1];

%word2charlist=();
open(INFILE, $word2charfile) || die("Can't open chinese word to char map: ".$word2charfile."\n");
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


open(INFILE, $tranfile) || die("Can't open transcription file ".$tranfile."\n");
while(<INFILE>) {
  chomp;
  @line = split(/\s+/);

  ## utt_id
  print $line[0];
  ## utt_character_word
  for($i=1; $i<scalar(@line); $i++) {
    if(!exists($word2charlist{$line[$i]})) {
      print " ".$line[$i];
    }
    else {
      print $word2charlist{$line[$i]};
    }
  }
  print "\n";
}
close(INFILE);

