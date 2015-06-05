#!/usr/bin/env perl

# Apache 2.0.
# Guoguo Chen (guoguo@jhu.edu)
# Daniel Povey (dpovey@gmail.com)
#
# Validation script for data/local/dict


if(@ARGV != 1) {
  die "Usage: validate_dict_dir.pl dict_directory\n";
}

$dict = shift @ARGV;
$dict =~ s:/$::;

$exit = 0;
$success = 1;  # this is re-set each time we read a file.

sub set_to_fail { $exit = 1; $success = 0; }

# Checking silence_phones.txt -------------------------------
print "Checking $dict/silence_phones.txt ...\n";
if(-z "$dict/silence_phones.txt") {print "--> ERROR: $dict/silence_phones.txt is empty or not exists\n"; exit 1;}
if(!open(S, "<$dict/silence_phones.txt")) {print "--> ERROR: fail to open $dict/silence_phones.txt\n"; exit 1;}
$idx = 1;
%silence = ();

print "--> reading $dict/silence_phones.txt\n";
while(<S>) {
  if (! s/\n$//) {
    print "--> ERROR: last line '$_' of $dict/silence_phones.txt does not end in newline.\n";
    set_to_fail();
  }
  my @col = split(" ", $_);
  if (@col == 0) {
    set_to_fail(); 
    print "--> ERROR: empty line in $dict/silence_phones.txt (line $idx)\n"; 
  }
  foreach(0 .. @col-1) {
    my $p = $col[$_];
    if($silence{$p}) {set_to_fail(); print "--> ERROR: phone \"$p\" duplicates in $dict/silence_phones.txt (line $idx)\n"; }
    else {$silence{$p} = 1;}
    if ($p =~ m/_$/ || $p =~ m/#/ || $p =~ m/_[BESI]$/){
      set_to_fail();
      print "--> ERROR: phone \"$p\" has disallowed written form";
      
    }
  }
  $idx ++;
}
close(S);
$success == 0 || print "--> $dict/silence_phones.txt is OK\n";
print "\n";

# Checking optional_silence.txt -------------------------------
print "Checking $dict/optional_silence.txt ...\n";
if(-z "$dict/optional_silence.txt") {print "--> ERROR: $dict/optional_silence.txt is empty or not exists\n"; exit 1;}
if(!open(OS, "<$dict/optional_silence.txt")) {print "--> ERROR: fail to open $dict/optional_silence.txt\n"; exit 1;}
$idx = 1;
$success = 1;
print "--> reading $dict/optional_silence.txt\n";
while(<OS>) {
  chomp;
  my @col = split(" ", $_);
  if ($idx > 1 or @col > 1) {
    set_to_fail(); print "--> ERROR: only 1 phone expected in $dict/optional_silence.txt\n"; 
  } elsif (!$silence{$col[0]}) {
    set_to_fail(); print "--> ERROR: phone $col[0] not found in $dict/silence_phones.txt\n"; 
  }
  $idx ++;
}
close(OS);
$success == 0 || print "--> $dict/optional_silence.txt is OK\n";
print "\n";

# Checking nonsilence_phones.txt -------------------------------
print "Checking $dict/nonsilence_phones.txt ...\n";
if(-z "$dict/nonsilence_phones.txt") {print "--> ERROR: $dict/nonsilence_phones.txt is empty or not exists\n"; exit 1;}
if(!open(NS, "<$dict/nonsilence_phones.txt")) {print "--> ERROR: fail to open $dict/nonsilence_phones.txt\n"; exit 1;}
$idx = 1;
%nonsilence = ();
$success = 1;
print "--> reading $dict/nonsilence_phones.txt\n";
while(<NS>) {
  if (! s/\n$//) {
    print "--> ERROR: last line '$_' of $dict/nonsilence_phones.txt does not end in newline.\n";
    set_to_fail();
  }
  my @col = split(" ", $_);
  if (@col == 0) {
    set_to_fail(); 
    print "--> ERROR: empty line in $dict/nonsilence_phones.txt (line $idx)\n"; 
  }
  foreach(0 .. @col-1) {
    my $p = $col[$_];
    if($nonsilence{$p}) {set_to_fail(); print "--> ERROR: phone \"$p\" duplicates in $dict/nonsilence_phones.txt (line $idx)\n"; }
    else {$nonsilence{$p} = 1;}
    if ($p =~ m/_$/ || $p =~ m/#/ || $p =~ m/_[BESI]$/){
      set_to_fail();
      print "--> ERROR: phone \"$p\" has disallowed written form";
      
    }
  }
  $idx ++;
}
close(NS);
$success == 0 || print "--> $dict/nonsilence_phones.txt is OK\n";
print "\n";

# Checking disjoint -------------------------------
sub intersect {
  my ($a, $b) = @_;
  @itset = ();
  %itset = ();
  foreach(keys %$a) {
    if(exists $b->{$_} and !$itset{$_}) {
      push(@itset, $_);
      $itset{$_} = 1;
    }
  }
  return @itset;
}

print "Checking disjoint: silence_phones.txt, nonsilence_phones.txt\n";
@itset = intersect(\%silence, \%nonsilence);
if(@itset == 0) {print "--> disjoint property is OK.\n";}
else {set_to_fail(); print "--> ERROR: silence_phones.txt and nonsilence_phones.txt has overlap: "; foreach(@itset) {print "$_ ";} print "\n";}
print "\n";


sub check_lexicon {
  my ($lex, $num_prob_cols, $num_skipped_cols) = @_;
  print "Checking $lex\n";
  !open(L, "<$lex") && print "--> ERROR: fail to open $lex\n" && set_to_fail();
  my %seen_line = {};
  $idx = 1; $success = 1;
  print "--> reading $lex\n";
  while (<L>) {
    if (defined $seen_line{$_}) {
      print "--> ERROR: line '$_' of $lex is repeated\n";
      set_to_fail();
    }
    $seen_line{$_} = 1;
    if (! s/\n$//) {
      print "--> ERROR: last line '$_' of $lex does not end in newline.\n";
      set_to_fail();
    }
    my @col = split(" ", $_);
    $word = shift @col;
    if (!defined $word) {
      print "--> ERROR: empty lexicon line in $lex\n"; set_to_fail();
    }
    if ($word eq "<s>" || $word eq "</s>") {
      print "--> ERROR: lexicon.txt contains forbidden word $word\n";
      set_to_fail();
    }
    for ($n = 0; $n < $num_prob_cols; $n++) {
      $prob = shift @col;
      if (!($prob > 0.0 && $prob <= 1.0)) { 
        print "--> ERROR: bad pron-prob in lexicon-line '$_', in $lex\n";
        set_to_fail();
      }
    }
    for ($n = 0; $n < $num_skipped_cols; $n++) { shift @col; }
    if (@col == 0) {
      print "--> ERROR: lexicon.txt contains word $word with empty ";
      print "pronunciation.\n";
      set_to_fail();
    }
    foreach (0 .. @col-1) {
      if (!$silence{@col[$_]} and !$nonsilence{@col[$_]}) {
        print "--> ERROR: phone \"@col[$_]\" is not in {, non}silence.txt ";
        print "(line $idx)\n"; 
        set_to_fail();
      }
    }
    $idx ++;
  }
  close(L);
  $success == 0 || print "--> $lex is OK\n";
  print "\n";
}

if (-f "$dict/lexicon.txt") { check_lexicon("$dict/lexicon.txt", 0, 0); }
if (-f "$dict/lexiconp.txt") { check_lexicon("$dict/lexiconp.txt", 1, 0); }
if (-f "$dict/lexiconp_silprob.txt") {
  # If $dict/lexiconp_silprob.txt exists, we expect $dict/silprob.txt to also
  # exist.
  check_lexicon("$dict/lexiconp_silprob.txt", 2, 2);
  if (-f "$dict/silprob.txt") {
    !open(SP, "<$dict/silprob.txt") &&
      print "--> ERROR: fail to open $dict/silprob.txt\n" && set_to_fail();
    while (<SP>) {
      chomp; my @col = split;
      @col != 2 && die "--> ERROR: bad line \"$_\"\n" && set_to_fail();
      if ($col[0] eq "<s>" || $col[0] eq "overall") {
        if (!($col[1] > 0.0 && $col[1] <= 1.0)) { 
          set_to_fail();
          print "--> ERROR: bad probability in $dir/silprob.txt \"$_\"\n";
        }
      } elsif ($col[0] eq "</s>_s" || $col[0] eq "</s>_n") {
        if ($col[1] <= 0.0) { 
          set_to_fail();
          print "--> ERROR: bad correction term in $dir/silprob.txt \"$_\"\n";
        }
      } else {
        print "--> ERROR: unexpected line in $dir/silprob.txt \"$_\"\n";
        set_to_fail();
      }
    }
    close(SP);
  } else {
    set_to_fail();
    print "--> ERROR: expecting $dict/silprob.txt to exist\n";
  }
}

if (!(-f "$dict/lexicon.txt" || -f "$dict/lexiconp.txt")) {
  print "--> ERROR: neither lexicon.txt or lexiconp.txt exist in directory $dir\n";
  set_to_fail();
}

sub check_lexicon_pair {
  my ($lex1, $num_prob_cols1, $num_skipped_cols1,
      $lex2, $num_prob_cols2, $num_skipped_cols2) = @_;
  # We have checked individual lexicons already.
  open(L1, "<$lex1"); open(L2, "<$lex2");
  print "Checking lexicon pair $lex1 and $lex2\n";
  my $line_num = 0;
  while(<L1>) {
    $line_num++;
    @A = split;
    $line_B = <L2>;
    if (!defined $line_B) {
      print "--> ERROR: $lex1 and $lex2 have different number of lines.\n";
      set_to_fail(); last;
    }
    @B = split(" ", $line_B);
    # Check if the word matches.
    if ($A[0] ne $B[0]) {
      print "--> ERROR: $lex1 and $lex2 mismatch at line $line_num. sorting?\n";
      set_to_fail(); last;
    }
    shift @A; shift @B;
    for ($n = 0; $n < $num_prob_cols1 + $num_skipped_cols1; $n ++) { shift @A; }
    for ($n = 0; $n < $num_prob_cols2 + $num_skipped_cols2; $n ++) { shift @B; }
    # Check if the pronunciation matches
    if (join(" ", @A) ne join(" ", @B)) {
      print "--> ERROR: $lex1 and $lex2 mismatch at line $line_num. sorting?\n";
      set_to_fail(); last;
    }
  }
  $line_B = <L2>;
  if (defined $line_B && $exit == 0) {
    print "--> ERROR: $lex1 and $lex2 have different number of lines.\n";
    set_to_fail();
  }
  $success == 0 || print "--> lexicon pair $lex1 and $lex2 match\n\n";
}

# If more than one lexicon exist, we have to check if they correspond to each
# other. It could be that the user overwrote one and we need to regenerate the
# other, but we do not know which is which.
if ( -f "$dict/lexicon.txt" && -f "$dict/lexiconp.txt") {
  check_lexicon_pair("$dict/lexicon.txt", 0, 0, "$dict/lexiconp.txt", 1, 0);
}
if ( -f "$dict/lexiconp.txt" && -f "$dict/lexiconp_silprob.txt") {
  check_lexicon_pair("$dict/lexiconp.txt", 1, 0,
                     "$dict/lexiconp_silprob.txt", 2, 2);
}

# Checking extra_questions.txt -------------------------------
%distinguished = (); # Keep track of all phone-pairs including nonsilence that
                     # are distinguished (split apart) by extra_questions.txt,
                     # as $distinguished{$p1,$p2} = 1.  This will be used to
                     # make sure that we don't have pairs of phones on the same
                     # line in nonsilence_phones.txt that can never be
                     # distinguished from each other by questions.  (If any two
                     # phones appear on the same line in nonsilence_phones.txt,
                     # they share a tree root, and since the automatic
                     # question-building treats all phones that appear on the
                     # same line of nonsilence_phones.txt as being in the same
                     # group, we can never distinguish them without resorting to
                     # questions in extra_questions.txt.
print "Checking $dict/extra_questions.txt ...\n";
if (-s "$dict/extra_questions.txt") {
  if (!open(EX, "<$dict/extra_questions.txt")) {
    set_to_fail(); print "--> ERROR: fail to open $dict/extra_questions.txt\n";
  }
  $idx = 1;
  $success = 1;
  print "--> reading $dict/extra_questions.txt\n";
  while(<EX>) {
    if (! s/\n$//) {
      print "--> ERROR: last line '$_' of $dict/extra_questions.txt does not end in newline.\n";
      set_to_fail();
    }
    my @col = split(" ", $_);
    if (@col == 0) {
      set_to_fail();  print "--> ERROR: empty line in $dict/extra_questions.txt\n";
    }
    foreach (0 .. @col-1) {
      if(!$silence{@col[$_]} and !$nonsilence{@col[$_]}) {
        set_to_fail();  print "--> ERROR: phone \"@col[$_]\" is not in {, non}silence.txt (line $idx, block ", $_+1, ")\n"; 
      }
      $idx ++;
    }
    %col_hash = ();
    foreach $p (@col) { $col_hash{$p} = 1; }
    foreach $p1 (@col) {
      # Update %distinguished hash.
      foreach $p2 (keys %nonsilence) {
        if (!defined $col_hash{$p2}) { # for each p1 in this question and p2 not
                                       # in this question (and in nonsilence
                                       # phones)... mark p1,p2 as being split apart
          $distinguished{$p1,$p2} = 1;
          $distinguished{$p2,$p1} = 1;
        }
      }
    }
  }
  close(EX);
  $success == 0 || print "--> $dict/extra_questions.txt is OK\n";
} else { print "--> $dict/extra_questions.txt is empty (this is OK)\n";}


# check nonsilence_phones.txt again for phone-pairs that are never
# distnguishable.  (note: this situation is normal and expected for silence
# phones, so we don't check it.)
if(!open(NS, "<$dict/nonsilence_phones.txt")) {
  print "--> ERROR: fail to open $dict/nonsilence_phones.txt the second time\n"; exit 1;
}

$num_warn_nosplit = 0;
$num_warn_nosplit_limit = 10;
while(<NS>) {
  my @col = split(" ", $_);
  foreach $p1 (@col) { 
    foreach $p2 (@col) {
      if ($p1 ne $p2 && ! $distinguished{$p1,$p2}) {
        set_to_fail();
        if ($num_warn_nosplit <= $num_warn_nosplit_limit) {
          print "--> ERROR: phones $p1 and $p2 share a tree root but can never be distinguished by extra_questions.txt.\n";
        }
        if ($num_warn_nosplit == $num_warn_nosplit_limit) {
          print "... Not warning any more times about this issue.\n";
        }
        if ($num_warn_nosplit == 0) {
          print "    (note: we started checking for this only recently.  You can still build a system but\n";
          print "     phones $p1 and $p2 will be acoustically indistinguishable).\n";
        }
        $num_warn_nosplit++;
      }
    }
  }
}


if ($exit == 1) {
  print "--> ERROR validating dictionary directory $dict (see detailed error ";
  print "messages above)\n\n";
  exit 1;
} else {
  print "--> SUCCESS [validating dictionary directory $dict]\n\n";
}

exit 0;
