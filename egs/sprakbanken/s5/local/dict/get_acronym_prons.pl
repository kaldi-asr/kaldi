#!/usr/bin/perl

# Reads a dictionary (for prons of letters), and an OOV list,
# and puts out candidate pronunciations of words in that list
# that could plausibly be acronyms.
# We judge that a word can plausibly be an acronym if it is
# a sequence of just letters (no non-letter characters such
# as "'"),  or something like U.K.,
# and the number of letters is four or less.
#
# If the text were not already pre-normalized, there would
# be other hints such as capitalization.

# This program appends
# the prons of the individual letters (A., B. and so on) to work out
# the pron of the acronym.
# Note: this is somewhat dependent on the convention used in CMUduct, that
# the individual letters are spelled this way (e.g. "A."). [it seems
# to also have the separated versions.

if (!(@ARGV == 1 || @ARGV == 2)) { 
  print "Usage: get_acronym_prons.pl dict [oovlist]";
}

$max_length = 4; # Max #letters in an acronym. (Longer 
 # acronyms tend to have "pseudo-pronunciations", e.g. think about UNICEF.

$dict = shift @ARGV;
open(D, "<$dict") || die "Opening dictionary $dict";

while(<D>) { # Read the dict, to get the prons of the letters.
  chop;
  @A = split(" ", $_);
  $word = shift @A;
  $pron = join(" ", @A);
  if ($word =~ m/^([A-Z])\.$/ ) {
    chop $word; # Remove trailing "." to get just the letter
    $letter = $1;
    if (!defined $letter_prons{$letter} ) { 
      $letter_prons{$letter} = [ ]; # new anonymous array
    }
    $arrayref = $letter_prons{$letter};
    push @$arrayref, $pron;
  } elsif( length($word) <= $max_length ) {
    $pronof{$word . "," . $pron} = 1;
    $isword{$word} = 1;
    #if (!defined $prons{$word} ) {
    #  $prons{$word} = [ ];
    #}
    #  push @{$prons{$word}}, $pron;
  }
}

sub get_letter_prons;

while(<>) { # Read OOVs.
  # For now, just do the simple cases without "." in 
  # between... things with "." in the OOV list seem to
  # be mostly errors.
  chop;
  $word = $_;
  if ($word =~ m/^[A-Z]{1,5}$/) {
    foreach $pron ( get_letter_prons($word) ) { # E.g. UNPO
      print "$word  $pron\n";
    }
  } elsif ($word =~ m:^(\w\.){1,4}\w\.?$:) { # E.g. U.K.  Make the final "." optional.
    $letters = $word;
    $letters =~ s:\.::g;
    foreach $pron ( get_letter_prons($letters) ) { 
      print "$word  $pron\n";
    }
  }
}

sub get_letter_prons {
  @acronym = split("", shift); # The letters in the word.
  my @prons = ( "" );
  
  while (@acronym > 0) {
    $l = shift @acronym;
    $n = 1; # num-repeats of letter $l.
    while (@acronym > 0 && $acronym[0] eq $l) {
      $n++;
      shift @acronym;
    }
    my $arrayref = $letter_prons{$l};
    my @prons_of_block = ();
    if ($n == 1) { # Just one repeat.
      foreach $lpron ( @$arrayref ) {
        push @prons_of_block, $lpron; # typically (always?) just one pron of a letter.
      }
    } elsif ($n == 2) { # Two repeats.  Can be "double a" or "a a"
      foreach $lpron ( @$arrayref ) {
        push @prons_of_block, "D AH1 B AH0 L " . $lpron;
        push @prons_of_block, $lpron . " " . $lpron;
      }
    } elsif ($n == 3) { # can be "triple a" or "a a a"
      foreach $lpron ( @$arrayref ) {
        push @prons_of_block, "T R IH1 P AH0 L " . $lpron;
        push @prons_of_block, "$lpron $lpron $lpron";
      }
    } elsif ($n >= 4) { # let's say it can only be that letter repeated $n times..
      # not sure really.
      foreach $lpron ( @$arrayref ) {
        $nlpron = $lpron;
        for ($m = 1; $m < $n; $m++) { $nlpron = $nlpron . " " . $lpron; }
        push @prons_of_block, $nlpron;
      }
    }
    my @new_prons = ();
    foreach $pron (@prons) {
      foreach $pron_of_block(@prons_of_block) {
        if ($pron eq "") {
          push @new_prons, $pron_of_block;
        } else {
          push @new_prons, $pron . " " . $pron_of_block;
        }
      }
    }
    @prons = @new_prons;
  }
  return @prons;
}
