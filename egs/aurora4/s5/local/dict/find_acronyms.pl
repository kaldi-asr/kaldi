#!/usr/bin/perl

# Reads a dictionary, and prints out a list of words that seem to be pronounced
# as acronyms (not including plurals of acronyms, just acronyms).  Uses
# the prons of the individual letters (A., B. and so on) to judge this.
# Note: this is somewhat dependent on the convention used in CMUduct, that
# the individual letters are spelled this way (e.g. "A.").

$max_length = 6; # Max length of words that might be
 # acronyms.

while(<>) { # Read the dict.
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

foreach $word (keys %isword) {
  my @letter_prons = get_letter_prons($word);
  foreach $pron (@letter_prons) {
    if (defined $pronof{$word.",".$pron}) {
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
        push @prons_of_block, $lpron . $lpron;
      }
    } elsif ($n == 3) { # can be "triple a" or "a a a"
      foreach $lpron ( @$arrayref ) {
        push @prons_of_block, "T R IH1 P AH0 L " . $lpron;
        push @prons_of_block, $lpron . $lpron . $lpron;
      }
    } elsif ($n >= 4) { # let's say it can only be that letter repeated $n times..
      # not sure really.
      foreach $lpron ( @$arrayref ) {
        $nlpron = "";
        for ($m = 0; $m < $n; $m++) { $nlpron = $nlpron . $lpron; }
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
