#!/usr/bin/perl
# Copyright Hong Kong University of Science and Technology (Author: Ricky Chan) 2013.
# 
# A script for dictionary generation with an input dict and a wordlist 
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

if($#ARGV+1 != 2 && $#ARGV+1 != 3) {
  printUsage();
  exit;
}

$usespron=0;
if(@ARGV == 3) {
  if($ARGV[2] ne "--spron") {
    printUsage();
    exit;
  }
  $usespron=1;
}

$dictfile=$ARGV[0];
$inputfile=$ARGV[1];

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
  @line = split(/\s+/);

  ## single pronunciation handling
  if($usespron==1) {
    if(scalar(@line)==0) {
      next;
    }

    for($i=0; $i<scalar(@line); $i++) {
      print $line[$i]." ";
    }
    print "\t";

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
    print "usage: perl hkust_extract_subdict.pl dict wordlist [--spron]\n\n";
    print "### this script handle multiple pronunciations for dict in default\n";
    print "### if you want to extract single(top) pronunciation from dict, please use the option --spron\n\n";
}

