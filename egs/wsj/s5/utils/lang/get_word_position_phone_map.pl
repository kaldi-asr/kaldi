#!/usr/bin/env perl

# Copyright 2018  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0.
#
use strict;
use warnings;

my $Usage = <<EOU;
# This script is for creating a mapping from word-position-dependent phones
# (with _I, _B, _E, _S suffixes) to word-position-independent phones,
# along with a word-position-independent version of phones.txt.
# It should only be required in very unusual situations.

Usage: utils/lang/get_word_position_phone_map.pl <lang-dir> <output-dir>

<lang-dir> is a conventional lang dir as validated by validate_lang.pl.
It is an error if <lang-dir> does not have word-position-dependent phones.

To <output-dir> will be written the following files:
  phones.txt is a conventional symbol table, similar in format to the one
   in <lang-dir>, but without word-position-dependency or disambiguation
   symbols.
  phone_map.int is a mapping from the input <lang-dir>'s phones to
   the phones in <output-dir>/phones.txt, containing integers, i.e.
   <word-position-dependent-phone> <word-position-independent-phone>.
  phone_map.txt is the text form of the mapping in phone_map.int, mostly
   provided for reference.
EOU


if (@ARGV != 2) {
  die $Usage;
}

my $lang_dir = shift @ARGV;
my $output_dir = shift @ARGV;

foreach my $filename ( ("phones.txt", "phones/disambig.int") ) {
  if (! -f "$lang_dir/$filename") {
    die "$0: expected file $lang_dir/$filename to exist";
  }
}

if (! -d $output_dir) {
  die "$0: expected directory $output_dir to exist";
}


# %is_disambig is a hash indexed by integer phone index in the input $lang_dir,
# which will contain 1 for each (integer) disambiguation symbol.
my %is_disambig;

open(D, "<$lang_dir/phones/disambig.int") || die "opening $lang_dir/phones/disambig.int";
while (<D>) {
  my $disambig_sym = int($_);
  $is_disambig{$disambig_sym} = 1;
}
close(D);

## @orig_phone_list will be an array indexed by integer index, containing
## the written form of the original, non-word-position-dependent phones.
## (but excluding disambiguation symbols like #0, #1 and so on).
## E.g. @orig_phone_list = ( "<eps>", "SIL", "SIL_B", "SIL_E", "SIL_I", "SIL_S", ... )
my @orig_phone_list = ();


## @mapped_phones will be an array of the same size as @orig_phone_list, but
## containing the same phone mapped to context-independent form,
## e.g. ( "<eps>", "SIL", "SIL", "SIL", SIL", "SIL",... )
my @mapped_phones = ();


## @mapped_phone_list will contain the distinct mapped phones in order,
## e.g. ( "<eps>", "SIL", "AA", ... )
my @mapped_phone_list = ();

## mapped_phone_to_int will be a mapping from the strings in @mapped_phones,
## such as "<eps>" and "SIL", to an integer like 0, 1, ....
my %mapped_phone_to_int;

# $cur_mapped_int keeps track of the symbols we've used in the output
# phones.txt.
my $cur_mapped_int = 0;

# $cur_line is the current line index in input phones.txt
my $cur_line = 0;

open(F, "<$lang_dir/phones.txt") || die "$0: failed to open $lang_dir/phones.txt for reading";

while (<F>) {
  chop;  # remove newline from $_ (the line we just read) for easy printing.
  my @A = split;  # split $_ on space.
  if (@A != 2) {  # if the array @A does not have length 2...
    die "$0: bad line $_ in file $lang_dir/phones.txt";
  }
  my $phone_name = $A[0];  # e.g. "<eps>" or "SIL" or "SIL_B" ...
  my $phone_int = int($A[1]);
  if ($phone_int != $cur_line) {
    die ("$0: unexpected line $_ in $lang_dir/phones.txt, expected integer to be $cur_line");
  }
  if (! $is_disambig{$phone_int}) {
    # if it's not a disambiguation symbol...
    my $mapped_phone_name = $phone_name;
    $mapped_phone_name =~ s/_[BESI]$//;

    push @orig_phone_list, $phone_name;
    push @mapped_phones, $mapped_phone_name;

    if (!defined $mapped_phone_to_int{$mapped_phone_name}) {
      $mapped_phone_to_int{$mapped_phone_name} = $cur_mapped_int++;
      push @mapped_phone_list, $mapped_phone_name;
    }
  }
  $cur_line++;
}
close(F);

if ($cur_line == 0) {
  die "$0: empty $lang_dir/phones.txt";
}

if ($cur_mapped_int == @orig_phone_list) {
  # if the number of distinct mapped phones is the same as the
  # number of input phones (including epsilon), it means the mapping
  # was a no-op.  This is an error, because it doesn't make sense to
  # run this script on input that was not word-position-dependent.
  die "input lang dir $lang_dir was not word-position-dependent.";
}

open(P, ">$output_dir/phones.txt") || die "failed to open $output_dir/phones.txt for writing.";
open(I, ">$output_dir/phone_map.int") || die "failed to open $output_dir/phone_map.int for writing.";
open(T, ">$output_dir/phone_map.txt") || die "failed to open $output_dir/phone_map.txt for writing.";

for (my $x = 0; $x <= $#mapped_phone_list; $x++) {
  print P "$mapped_phone_list[$x] $x\n";
}


for (my $x = 0; $x <= $#orig_phone_list; $x++) {
  my $orig_phone_name = $orig_phone_list[$x];
  my $mapped_phone_name = $mapped_phones[$x];
  my $y = $mapped_phone_to_int{$mapped_phone_name};
  defined $y || die "code error";

  print I "$x $y\n";
  print T "$orig_phone_name $mapped_phone_name\n";
}


(close(I) && close(T) && close(P)) || die "failed to close file (disk full?)";


exit(0);
