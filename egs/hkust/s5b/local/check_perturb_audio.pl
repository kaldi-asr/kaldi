#!/usr/bin/perl

# Copyright 2014. Ricky Chan Ho Yin
#
# This script compares the test_scp and train_scp (and file size & content of 
# the corresponding audio files (e.g. wav files) in --deep mode) to check if the 
# test_scp (test set files) actually holdout from the train_scp (train set files)
#
# train_scp/test_scp format as follow:
# id	filename
# 

use File::Compare;

if(@ARGV != 2 && @ARGV != 3) {
  printUsage();
  exit;
}

$deep_check=0;

if(@ARGV == 2) {
  $trainscp=$ARGV[0];
  $testscp=$ARGV[1];
}
else {
  if($ARGV[0] ne "--deep") {
    printUsage();
    exit;
  }
  $deep_check=1;
  $trainscp=$ARGV[1];
  $testscp=$ARGV[2];
}

%trainlist=();
%testlist=();

%trainlistlen=();
%testlistlen=();

open(INFILE, $trainscp) || die("Can't open train_scp ".$trainscp."\n");
while(<INFILE>) {
  chomp;
  @line=split(/\s+/);
  $a=$line[0];
  $b=$line[1];
  $trainlist{$b}=$a;
  if($deep_check) {
    $tmp = -s $b;
    push( @{$trainlistlen{$tmp}}, $b ); # use length for identification
  }
}
close(INFILE);

open(INFILE, $testscp) || die("Can't open test_scp ".$testscp."\n");
while(<INFILE>) {
  chomp;
  @line=split(/\s+/);
  $a=$line[0];
  $b=$line[1];
  $testlist{$b}=$a;
  if($deep_check) {
    $tmp = -s $b;
    $testlistlen{$b}=$tmp;  # file name to file size hash
  }
}
close(INFILE);

$totprocess=0;
$totfail=0;

for $key ( keys %testlist ) {
  if($totprocess%100 eq 0) {
    print "Total processed files: ".$totprocess."\n";
  }
  $value=$testlist{$key};
  if(exists($trainlist{$key})) {
    print "Warning: perturbed file found (test_scp file found in train_scp file)=> filename: ".$key." id: ".$value."\n";
    $totfail++;
    $totprocess++;
    next;
  }

  if($deep_check) {
    $length = $testlistlen{$key};
    if(exists($trainlistlen{$length})) {
      foreach $ref (sort @{$trainlistlen{$length}}) {
        if(compare($key, $ref)==0) {
          print "Warning: perturbed file found=> test_scp file: ".$key." and train_scp file: ".$ref." are identical\n";
          $totfail++;
        }
      }
    }
  }
  $totprocess++;
}

print "\nSummary:\n";
print "Total processed files: ".$totprocess."\n";
print "Total fail files: ".$totfail."\n";

sub printUsage() {
  print "Usage: perl check_perturb_audio.pl [--deep] train_scp test_scp\n";
  print "e.g.:  perl check_perturb_audio.pl --deep train_wav.scp test_wav.scp\n";
}

