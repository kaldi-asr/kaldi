#!/usr/bin/perl
# Copyright Hong Kong University of Science and Technology (Author: Ricky Chan) 2013.
# 
# A script for dictionary generation with a input dict and wordlist 
#
# example of dict format as follow:
# WORD1 ph1 ph2
# WORD2 ph1 ph2 ph3
# WORDX ph4
# WORDY ph4 ph5
# WORDZ ph3 ph1
#
# example of wordlist (support phrase of words) format as follow:
# WORD1
# WORD2
# WORDX WORDY 
# WORDX WORDY WORDZ

if(@ARGV < 2 || @ARGV > 4) {
  printUsage();
  exit;
}

$dictfile = shift @ARGV;
$inputfile = shift @ARGV;

$usespron=0;
$mergeword=0;
$mergewordhypen=0;

while (@ARGV > 0) {
  $param = shift @ARGV;
  if($param eq "--spron") { $usespron=1; }
  elsif ($param eq "--mergewords" ) { $mergeword = 1; }
  elsif ($param eq "--mergewords_withhypen" ) { $mergewordhypen = 1; }
  else { printUsage(); exit; }
}

if($mergeword==1 && $mergewordhypen==1) {
  print "--mergewords option and --mergewords_withhypen option can not be used at the same time,\n";
  print "please apply with only one of them.\n";
  exit;
}

%dictionarylist=();
open(INFILE, $dictfile) || die("Can't open dict ".$dictfile."\n");
while(<INFILE>){
  chomp;
  @line=split(/\s+/);
  $a=$line[0];
  $b="";
  for($i=1; $i<scalar(@line); $i++) {
    $b=$b . " " . $line[$i];
  }
  push ( @{ $dictionarylist{$a} }, $b );
}
close(INFILE);

open(INFILE, $inputfile) || die("Can't open wordlist ".$inputfile."\n");
while(<INFILE>) {
  chomp;
  $phrase = $_;
  if($mergeword==1) {
    $phrase =~ s/\s+//g;
  }
  elsif($mergewordhypen==1) {
    $phrase =~ s/\s+/-/g;
  }
  @line = split(/\s+/);

  ## single pronunciation handling
  if($usespron==1) {
    if(scalar(@line)==0) {
      next;
    }

    print $phrase."\t";

    for($i=0; $i<scalar(@line); $i++) {
      if(!exists($dictionarylist{$line[$i]})) {
        print " _NOT_FOUND_";
      }
      else {
        @ref=@{ $dictionarylist{$line[$i]} };
        print $ref[0]."";
      }
    }
    print "\n";
    next;
  }

  ## multiple pronunciations handling 
  @pronlist=();
  @tmppronlist=();

  if(scalar(@line)>0) {
    $word = $line[$0];
    if(!exists($dictionarylist{$word})) {
        push(@pronlist, '_NOT_FOUND_');
    }
    else {
      @ref=@{ $dictionarylist{$word} };
      for($i=0; $i<scalar(@ref); $i++) {
        push(@pronlist, $ref[$i]."");
      }
    }

    for($i=1; $i<scalar(@line); $i++) {
      $word = $line[$i];
      if(!exists($dictionarylist{$word})) {
        for($j=0; $j<scalar(@pronlist); $j++) {
          $pronlist[$j] = $pronlist[$j]." _NOT_FOUND_";
        }
      }
      else {
        @ref=@{ $dictionarylist{$word} };
        while(scalar(@pronlist)>0) {
          push(@tmppronlist, shift(@pronlist));
        }
        while(scalar(@tmppronlist)>0) {
          $tmppron = shift(@tmppronlist);
          for($j=0; $j<scalar(@ref); $j++) {
            push(@pronlist, $tmppron." ".$ref[$j]);
          } 
        }
      }
    }
    
    for($i=0; $i<scalar(@pronlist); $i++) {
      print $phrase."\t".$pronlist[$i]."\n";
    }
  }

}
close(INFILE);

sub printUsage {
    print "usage: perl hkust_extract_subdict.pl dict wordlist [--spron] [--mergewords | --mergewords_withhypen]\n\n";
    print "### This script can output a subdict when a dictionary and a wordlist are supplied\n";
    print "### This script can also generate dict entries for wordlist with multiple words in line\n\n";
    print "### This script handles multiple pronunciations for dict by default.\n";
    print "### If you want to extract single(top) pronunciation from dict, please use the option --spron\n\n";
    print "### The --mergewords option is useful if you want to merge the multiple words to single phrase \n";
    print "    in output format (e.g. 特別 行政 區 => 特別行政區)\n";
    print "### The --mergewords_withhypen option is the same as --mergewords option except it merges the \n";
    print "    multiple words with hypen in between (e.g. MACBOOK PRO => MACBOOK-PRO) in output format\n\n";
}

