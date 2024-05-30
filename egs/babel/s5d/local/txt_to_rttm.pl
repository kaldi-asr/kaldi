#!/usr/bin/perl

use strict;
use warnings;
use Getopt::Long;

my $Usage = <<EOU;
Usage:    txt2rttm.pl [options] <txt_in|-> <rttm_out|->

Allowed options:
  --flen                      : Frame length (float, default = 0.1)
  --symtab                    : Symbol table (string, default = "")
  --segment                   : Segment file from Kaldi (string, default = "")
EOU

my $symtab = "";
my $segment = "";
my $flen = 0.01;
GetOptions('symtab=s'   => \$symtab,
           'segment=s'  => \$segment,
           'flen=f'     => \$flen);

if ($symtab) {
  if (!open(S, "<$symtab")) {print "Fail to open symbol table: $symtab\n"; exit 1;}
}

if ($segment) {
  if (!open(SEG, "<$segment")) {print "Fail to open segment file: $segment\n"; exit 1;}
}

if(@ARGV != 2) {
  die $Usage;
}

# Get parameters
my $filein = shift @ARGV;
my $fileout = shift @ARGV;

# Get input source
my $source = "";
if ($filein eq "-") {
  $source = "STDIN";
} else {
  if (!open(I, "<$filein")) {print "Fail to open input file: $filein\n"; exit 1;}
  $source = "I";
}

# Open output fst list
my $sourceout = "";
if ($fileout ne "-") {
  if (!open(O, ">$fileout")) {print "Fail to open output file: $fileout\n"; exit 1;}
  $sourceout = "O";
}

# Get symbol table and start time
my %sym = ();
my %tbeg = ();
my %uid2utt = ();
if ($symtab) {
  while(<S>) {
    chomp;
    my @col = split(" ", $_);
    @col == 2 || die "Bad number of columns in $symtab\n";
    $sym{$col[1]} = $col[0];
  }
}

if ($segment) {
  while(<SEG>) {
    chomp;
    my @col = split(" ", $_);
    @col == 4 || die "Bad number of columns in $segment\n";
    $tbeg{$col[0]} = $col[2];
    $uid2utt{$col[0]} = $col[1];
  }
}

# Processing
while (<$source>) {
  chomp;
  my @col = split(" ", $_);
  my $uid = shift @col;
  my $words = join(" ", @col);
  @col = split(/;/, $words);

  my $utt = $uid;
  my $sta = 0;
  if ($segment) {
    $utt = $uid2utt{$uid};
    $sta = $tbeg{$uid};
  }
  foreach (@col) {
    my @subcol = split(" ", $_);
    @subcol == 2 || die "Bad number of columns in word-frame pair\n";
    my $word = $subcol[0];
    my $dur = $subcol[1]*$flen;
    my $lex = "LEXEME";
    if ($symtab) {$word = $sym{$word};}
    if ($word =~ m/^<.*>$/) {$lex = "NON-LEX";}
    eval "print $sourceout \"$lex $utt 1 $sta $dur $word <NA> <NA> <NA>\n\"";
    $sta += $dur;
  }
}

if ($symtab)  {close(S);}
if ($segment) {close(SEG);}
if ($filein  ne "-") {close(I);}
if ($fileout ne "-") {close(O);}
