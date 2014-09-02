#!/usr/bin/perl

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
  my @row = split(" ", $_);
  if (@row == 0) {
    set_to_fail(); 
    print "--> ERROR: empty line in $dict/nonsilence_phones.txt (line $idx)\n"; 
  }
  foreach(0 .. @row-1) {
    my $p = $row[$_];
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
  my ($lexfn, $pron_probs) = @_;
  print "Checking $lexfn\n";
  if(-z "$lexfn") {set_to_fail(); print "--> ERROR: $lexfn is empty or not exists\n";}
  if(!open(L, "<$lexfn")) {set_to_fail(); print "--> ERROR: fail to open $lexfn\n";}
  $idx = 1;
  $success = 1;
  print "--> reading $lexfn\n";
  while (<L>) {
    if (! s/\n$//) {
      print "--> ERROR: last line '$_' of $lexfn does not end in newline.\n";
      set_to_fail();
    }
    my @row = split(" ", $_);
    $word = shift @row;
    if (!defined $word) {
      set_to_fail(); print "--> ERROR: empty lexicon line in $lexfn\n"; 
    }
    if ($pron_probs) {
      $prob = shift @row;
      if (!($prob > 0.0 && $prob <= 1.0)) { 
        set_to_fail(); print "--> ERROR: bad pron-prob in lexicon-line '$_', in $lexfn\n";
      }
    }
    foreach (0 .. @row-1) {
      if (!$silence{@row[$_]} and !$nonsilence{@row[$_]}) {
        set_to_fail(); print "--> ERROR: phone \"@row[$_]\" is not in {, non}silence.txt (line $idx)\n"; 
      }
    }
    $idx ++;
  }
  close(L);
  $success == 0 || print "--> $lexfn is OK\n";
  print "\n";
}

if (-f "$dict/lexicon.txt") { check_lexicon("$dict/lexicon.txt", 0); }
if (-f "$dict/lexiconp.txt") { check_lexicon("$dict/lexiconp.txt", 1); }
if (!(-f "$dict/lexicon.txt" || -f "$dict/lexiconp.txt")) {
  print "--> ERROR: neither lexicon.txt or lexiconp.txt exist in directory $dir\n";
  set_to_fail();
}
# If both lexicon.txt and lexiconp.txt exist, we check that they correspond to
# each other.  If not, it could be that the user overwrote one and we need to
# regenerate the other, but we don't know which is which.
if ( (-f "$dict/lexicon.txt") && (-f "$dict/lexiconp.txt")) {
  print "Checking that lexicon.txt and lexiconp.txt match\n";
  if (!open(L, "<$dict/lexicon.txt") || !open(P, "<$dict/lexiconp.txt")) {
    die "Error opening lexicon.txt and/or lexiconp.txt"; # already checked, so would be code error.
  }
  while(<L>) {
    if (! s/\n$//) {
      print "--> ERROR: last line '$_' of $dict/lexicon.txt does not end in newline.\n";
      set_to_fail();
      last;
    }
    @A = split;
    $x = <P>;
    if ($x !~ s/\n$//) {
      print "--> ERROR: last line '$x' of $dict/lexiconp.txt does not end in newline.\n";
      set_to_fail();
      last;
    }
    if (!defined $x) {
      print "--> ERROR: lexicon.txt and lexiconp.txt have different numbers of lines (mismatch); delete one.\n";
      set_to_fail();
      last;
    }
    @B = split(" ", $x);
    $w = shift @B;
    $p = shift @B;
    unshift @B, $w;
    # now @A and @B should be the same.
    if ($#A != $#B) {
      print "--> ERROR: lexicon.txt and lexiconp.txt have mismatched lines '$_' versus '$x'; delete one.\n";
      set_to_fail();
      last;
    }
    for ($n = 0; $n < @A; $n++) {
      if ($A[$n] ne $B[$n]) {
        print "--> ERROR: lexicon.txt and lexiconp.txt have mismatched lines '$_' versus '$x'; delete one.\n";
        set_to_fail();
        last;
      }
    }
  }
  $x = <P>;
  if (defined $x && $exit == 0) {
    print "--> ERROR: lexicon.txt and lexiconp.txt have different numbers of lines (mismatch); delete one.\n";
    set_to_fail();
  }
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
    my @row = split(" ", $_);
    if (@row == 0) {
      set_to_fail();  print "--> ERROR: empty line in $dict/extra_questions.txt\n";
    }
    foreach (0 .. @row-1) {
      if(!$silence{@row[$_]} and !$nonsilence{@row[$_]}) {
        set_to_fail();  print "--> ERROR: phone \"@row[$_]\" is not in {, non}silence.txt (line $idx, block ", $_+1, ")\n"; 
      }
      $idx ++;
    }
    %row_hash = ();
    foreach $p (@row) { $row_hash{$p} = 1; }
    foreach $p1 (@row) {
      # Update %distinguished hash.
      foreach $p2 (keys %nonsilence) {
        if (!defined $row_hash{$p2}) { # for each p1 in this question and p2 not
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
  my @row = split(" ", $_);
  foreach $p1 (@row) { 
    foreach $p2 (@row) {
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


if ($exit == 1) { print "--> ERROR validating dictionary directory $dict (see detailed error messages above)\n"; exit 1;}
else { print "--> SUCCESS [validating dictionary directory $dict]\n"; }

exit 0;
