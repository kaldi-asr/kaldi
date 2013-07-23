#!/usr/bin/perl
# Copyright Hong Kong University of Science and Technology (Author: Ricky Chan) 2013.

if($#ARGV+1 != 2) {
  print "usage: perl hkust_extract_subdict.pl dict wordlist \n";
  exit;
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
  $dictionarylist{$a}=$b;
}
close(INFILE);

open(INFILE, $inputfile) || die("Can't open wordlist ".$inputfile."\n");
while(<INFILE>) {
  chomp;
  @line = split(/\s+/);

  for($i=0; $i<scalar(@line); $i++) {
    print $line[$i]." ";
  }
  print "\t";

  for($i=0; $i<scalar(@line); $i++) {
    if(! exists($dictionarylist{$line[$i]})) {
      print "_NOT_FOUND_ ";
    }
    else {
      print $dictionarylist{$line[$i]}." ";
    }
  }
  print "\n";
}
close(INFILE);


