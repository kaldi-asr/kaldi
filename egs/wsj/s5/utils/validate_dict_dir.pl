#!/usr/bin/env perl

# Apache 2.0.
# Copyright  2012 Guoguo Chen
#            2015 Daniel Povey
#            2017 Johns Hopkins University (Jan "Yenda" Trmal <jtrmal@gmail.com>)
#
# Validation script for 'dict' directories (e.g. data/local/dict)

# this function reads the opened file (supplied as a first
# parameter) into an array of lines. For each
# line, it tests whether it's a valid utf-8 compatible
# line. If all lines are valid utf-8, it returns the lines
# decoded as utf-8, otherwise it assumes the file's encoding
# is one of those 1-byte encodings, such as ISO-8859-x
# or Windows CP-X.
# Please recall we do not really care about
# the actually encoding, we just need to
# make sure the length of the (decoded) string
# is correct (to make the output formatting looking right).
sub get_utf8_or_bytestream {
  use Encode qw(decode encode);
  my $is_utf_compatible = 1;
  my @unicode_lines;
  my @raw_lines;
  my $raw_text;
  my $lineno = 0;
  my $file = shift;

  while (<$file>) {
    $raw_text = $_;
    last unless $raw_text;
    if ($is_utf_compatible) {
      my $decoded_text = eval { decode("UTF-8", $raw_text, Encode::FB_CROAK) } ;
      $is_utf_compatible = $is_utf_compatible && defined($decoded_text);
      push @unicode_lines, $decoded_text;
    } else {
      #print STDERR "WARNING: the line $raw_text cannot be interpreted as UTF-8: $decoded_text\n";
      ;
    }
    push @raw_lines, $raw_text;
    $lineno += 1;
  }

  if (!$is_utf_compatible) {
    return (0, @raw_lines);
  } else {
    return (1, @unicode_lines);
  }
}

# check if the given unicode string contain unicode whitespaces
# other than the usual four: TAB, LF, CR and SPACE
sub validate_utf8_whitespaces {
  my $unicode_lines = shift;
  use feature 'unicode_strings';
  for (my $i = 0; $i < scalar @{$unicode_lines}; $i++) {
    my $current_line = $unicode_lines->[$i];
    if ((substr $current_line, -1) ne "\n"){
      print STDERR "$0: The current line (nr. $i) has invalid newline\n";
      return 1;
    }
    # we replace TAB, LF, CR, and SPACE
    # this is to simplify the test
    if ($current_line =~ /\x{000d}/) {
      print STDERR "$0: The current line (nr. $i) contains CR (0x0D) character\n";
      return 1;
    }
    $current_line =~ s/[\x{0009}\x{000a}\x{0020}]/./g;
    if ($current_line =~/\s/) {
      return 1;
    }
  }
  return 0;
}

# checks if the text in the file (supplied as the argument) is utf-8 compatible
# if yes, checks if it contains only allowed whitespaces. If no, then does not
# do anything. The function seeks to the original position in the file after
# reading the text.
sub check_allowed_whitespace {
  my $file = shift;
  my $pos = tell($file);
  (my $is_utf, my @lines) = get_utf8_or_bytestream($file);
  seek($file, $pos, SEEK_SET);
  if ($is_utf) {
    my $has_invalid_whitespaces = validate_utf8_whitespaces(\@lines);
    print "--> text seems to be UTF-8 or ASCII, checking whitespaces\n";
    if ($has_invalid_whitespaces) {
      print "--> ERROR: the text containes disallowed UTF-8 whitespace character(s)\n";
      return 0;
    } else {
      print "--> text contains only allowed whitespaces\n";
    }
  } else {
    print "--> text doesn't seem to be UTF-8 or ASCII, won't check whitespaces\n";
  }
  return 1;
}


if(@ARGV != 1) {
  die "Usage: validate_dict_dir.pl <dict-dir>\n" .
      "e.g.: validate_dict_dir.pl data/local/dict\n";
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
$crlf = 1;

print "--> reading $dict/silence_phones.txt\n";
check_allowed_whitespace(\*S) || set_to_fail();
while(<S>) {
  if (! s/\n$//) {
    print "--> ERROR: last line '$_' of $dict/silence_phones.txt does not end in newline.\n";
    set_to_fail();
  }
  if ($crlf == 1 && m/\r/) {
    print "--> ERROR: $dict/silence_phones.txt contains Carriage Return (^M) characters.\n";
    set_to_fail();
    $crlf = 0;
  }
  my @col = split(" ", $_);
  if (@col == 0) {
    set_to_fail();
    print "--> ERROR: empty line in $dict/silence_phones.txt (line $idx)\n";
  }
  foreach(0 .. @col-1) {
    my $p = $col[$_];
    if($silence{$p}) {
      set_to_fail(); print "--> ERROR: phone \"$p\" duplicates in $dict/silence_phones.txt (line $idx)\n";
    } else {
      $silence{$p} = 1;
    }
    # disambiguation symbols; phones ending in _B, _E, _S or _I will cause
    # problems with word-position-dependent systems, and <eps> is obviously
    # confusable with epsilon.
    if ($p =~ m/^#/ || $p =~ m/_[BESI]$/ || $p eq "<eps>"){
      set_to_fail();
      print "--> ERROR: phone \"$p\" has disallowed written form\n";
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
$crlf = 1;
print "--> reading $dict/optional_silence.txt\n";
check_allowed_whitespace(\*OS) or exit 1;
while(<OS>) {
  chomp;
  my @col = split(" ", $_);
  if ($idx > 1 or @col > 1) {
    set_to_fail(); print "--> ERROR: only 1 phone expected in $dict/optional_silence.txt\n";
  } elsif (!$silence{$col[0]}) {
    set_to_fail(); print "--> ERROR: phone $col[0] not found in $dict/silence_phones.txt\n";
  }
  if ($crlf == 1 && m/\r/) {
    print "--> ERROR: $dict/optional_silence.txt contains Carriage Return (^M) characters.\n";
    set_to_fail();
    $crlf = 0;
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
$crlf = 1;
print "--> reading $dict/nonsilence_phones.txt\n";
check_allowed_whitespace(\*NS) or set_to_fail();
while(<NS>) {
  if ($crlf == 1 && m/\r/) {
    print "--> ERROR: $dict/nonsilence_phones.txt contains Carriage Return (^M) characters.\n";
    set_to_fail();
    $crlf = 0;
  }
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
    if($nonsilence{$p}) {
      set_to_fail(); print "--> ERROR: phone \"$p\" duplicates in $dict/nonsilence_phones.txt (line $idx)\n";
    } else {
      $nonsilence{$p} = 1;
    }
    # phones that start with the pound sign/hash may be mistaken for
    # disambiguation symbols; phones ending in _B, _E, _S or _I will cause
    # problems with word-position-dependent systems, and <eps> is obviously
    # confusable with epsilon.
    if ($p =~ m/^#/ || $p =~ m/_[BESI]$/ || $p eq "<eps>"){
      set_to_fail();
      print "--> ERROR: phone \"$p\" has disallowed written form\n";
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
  $idx = 1; $success = 1; $crlf = 1;
  print "--> reading $lex\n";
  check_allowed_whitespace(\*L) or set_to_fail();
  while (<L>) {
    if ($crlf == 1 && m/\r/) {
      print "--> ERROR: $lex contains Carriage Return (^M) characters.\n";
      set_to_fail();
      $crlf = 0;
    }
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
    if ($word eq "<s>" || $word eq "</s>" || $word eq "<eps>" || $word eq "#0") {
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
      $crlf = 1;
    while (<SP>) {
      if ($crlf == 1 && m/\r/) {
        print "--> ERROR: $dict/silprob.txt contains Carriage Return (^M) characters.\n";
        set_to_fail();
        $crlf = 0;
      }
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
  $crlf = 1;
  print "--> reading $dict/extra_questions.txt\n";
  check_allowed_whitespace(\*EX) or set_to_fail();
  while(<EX>) {
    if ($crlf == 1 && m/\r/) {
      print "--> ERROR: $dict/extra_questions.txt contains Carriage Return (^M) characters.\n";
      set_to_fail();
      $crlf = 0;
    }
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
        set_to_fail();  print "--> ERROR: phone \"@col[$_]\" is not in {, non}silence_phones.txt (line $idx, block ", $_+1, ")\n";
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

if (-f "$dict/nonterminals.txt") {
  open(NT, "<$dict/nonterminals.txt") || die "opening $dict/nonterminals.txt";
  my %nonterminals = ();
  my $line_number = 1;
  while (<NT>) {
    chop;
    my @line = split(" ", $_);
    if (@line != 1 || ! m/^#nonterm:/ || defined $nonterminals{$line[0]}) {
      print "--> ERROR: bad (or duplicate) line $line_number: '$_' in $dict/nonterminals.txt\n"; exit 1;
    }
    $nonterminals{$line[0]} = 1;
    $line_number++;
  }
  print "--> $dict/nonterminals.txt is OK\n";
}


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
