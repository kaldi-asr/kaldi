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

$exit = 0;
# Checking silence_phones.txt -------------------------------
print "Checking $dict/silence_phones.txt ...\n";
if(-z "$dict/silence_phones.txt") {print "--> ERROR: $dict/silence_phones.txt is empty or not exists\n"; exit 1;}
if(!open(S, "<$dict/silence_phones.txt")) {print "--> ERROR: fail to open $dict/silence_phones.txt\n"; exit 1;}
$idx = 1;
%silence = ();
$success = 1;
print "--> reading $dict/silence_phones.txt\n";
while(<S>) {
  chomp;
  my @col = split(" ", $_);
  foreach(0 .. @col-1) {
    my $p = $col[$_];
    if($silence{$p}) {$exit = 1; print "--> ERROR: phone \"$p\" duplicates in $dict/silence_phones.txt (line $idx)\n"; $success = 0;}
    else {$silence{$p} = 1;}
    if ($p =~ m/_$/ || $p =~ m/#/ || $p =~ m/_[BESI]$/){
      $exit = 1;
      print "--> ERROR: phone \"$p\" has disallowed written form";
      $success = 0;
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
    $exit = 1; print "--> ERROR: only 1 phone expected in $dict/optional_silence.txt\n"; $success = 0;
  } elsif (!$silence{$col[0]}) {
    $exit = 1; print "--> ERROR: phone $col[0] not found in $dict/silence_phones.txt\n"; $success = 0;
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
  chomp;
  my @col = split(" ", $_);
  foreach(0 .. @col-1) {
    my $p = $col[$_];
    if($nonsilence{$p}) {$exit = 1; print "--> ERROR: phone \"$p\" duplicates in $dict/nonsilence_phones.txt (line $idx)\n"; $success = 0;}
    else {$nonsilence{$p} = 1;}
    if ($p =~ m/_$/ || $p =~ m/#/ || $p =~ m/_[BESI]$/){
      $exit = 1;
      print "--> ERROR: phone \"$p\" has disallowed written form";
      $success = 0;
    }
  }
  $idx ++;
}
close(NS);
$success == 0 || print "--> $dict/silence_phones.txt is OK\n";
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
else {$exit = 1; print "--> ERROR: silence_phones.txt and nonsilence_phones.txt has overlap: "; foreach(@itset) {print "$_ ";} print "\n";}
print "\n";


sub check_lexicon {
  my ($lexfn, $pron_probs) = @_;
  print "Checking $lexfn\n";
  if(-z "$lexfn") {$exit = 1; print "--> ERROR: $lexfn is empty or not exists\n";}
  if(!open(L, "<$lexfn")) {$exit = 1; print "--> ERROR: fail to open $lexfn\n";}
  $idx = 1;
  $success = 1;
  print "--> reading $lexfn\n";
  while (<L>) {
    chomp;
    my @col = split(" ", $_);
    $word = shift @col;
    if (!defined $word) {
      $exit = 1; print "--> ERROR: empty lexicon line in $lexfn\n"; 
      $success = 0;
    }
    if ($pron_probs) {
      $prob = shift @col;
      if (!($prob > 0.0 && $prob <= 1.0)) { 
        $exit = 1; print "--> ERROR: bad pron-prob in lexicon-line '$_', in $lexfn\n";
        $success = 0;
      }
    }
    foreach (0 .. @col-1) {
      if (!$silence{@col[$_]} and !$nonsilence{@col[$_]}) {
        $exit = 1; print "--> ERROR: phone \"@col[$_]\" is not in {, non}silence.txt (line $idx)\n"; 
        $success = 0;
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
  $exit = 1;
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
    @A = split;
    $x = <P>;
    if (!defined $x) {
      print "--> ERROR: lexicon.txt and lexiconp.txt have different numbers of lines (mismatch); delete one.\n";
      $exit = 1;
      last;
    }
    @B = split(" ", $x);
    $w = shift @B;
    $p = shift @B;
    unshift @B, $w;
    # now @A and @B should be the same.
    if ($#A != $#B) {
      print "--> ERROR: lexicon.txt and lexiconp.txt have mismatched lines '$_' versus '$x'; delete one.\n";
      $exit = 1;
      last;
    }
    for ($n = 0; $n < @A; $n++) {
      if ($A[$n] ne $B[$n]) {
        print "--> ERROR: lexicon.txt and lexiconp.txt have mismatched lines '$_' versus '$x'; delete one.\n";
        $exit = 1;
        last;
      }
    }
  }
  $x = <P>;
  if (defined $x && $exit == 0) {
    print "--> ERROR: lexicon.txt and lexiconp.txt have different numbers of lines (mismatch); delete one.\n";
    $exit = 1;
  }
}

# Checking extra_questions.txt -------------------------------
print "Checking $dict/extra_questions.txt ...\n";
if (-s "$dict/extra_questions.txt") {
  if(!open(EX, "<$dict/extra_questions.txt")) {$exit = 1; print "--> ERROR: fail to open $dict/extra_questions.txt\n";}
  $idx = 1;
  $success = 1;
  print "--> reading $dict/extra_questions.txt\n";
  while(<EX>) {
    chomp;
    my @col = split(" ", $_);
    foreach(0 .. @col-1) {
      if(!$silence{@col[$_]} and !$nonsilence{@col[$_]}) {
        $exit = 1; print "--> ERROR: phone \"@col[$_]\" is not in {, non}silence.txt (line $idx, block ", $_+1, ")\n"; 
        $success = 0;
      }
    }
    $idx ++;
  } 
  close(EX);
  $success == 0 || print "--> $dict/extra_questions.txt is OK\n";
} else { print "--> $dict/extra_questions.txt is empty (this is OK)\n";}

if($exit == 1) { print " [Error detected ]\n"; exit 1;}

exit 0;
