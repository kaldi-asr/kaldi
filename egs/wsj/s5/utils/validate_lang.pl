#!/usr/bin/perl

# Apache 2.0.
# Copyright  2012   Guoguo Chen
#            2014   Neil Nelson
#
# Validation script for data/lang

if (@ARGV != 1) {
  print "Usage: $0 <lang_directory>\n";
  print "e.g.:  $0 data/lang\n";
  exit(1);
}

$lang = shift @ARGV;
$exit = 0;
$warning = 0;
# Checking phones.txt -------------------------------
print "Checking $lang/phones.txt ...\n";
if (-z "$lang/phones.txt") {print "--> ERROR: $lang/phones.txt is empty or does not exist\n"; exit 1;}
if (!open(P, "<$lang/phones.txt")) {print "--> ERROR: fail to open $lang/phones.txt\n"; exit 1;}
$idx = 1;
%psymtab = ();
while(<P>) {
  chomp;
  my @col = split(" ", $_);
  if (@col != 2) {print "--> ERROR: expect 2 columns in $lang/phones.txt (break at line $idx)\n"; exit 1;}
  my $phone = shift @col;
  my $id = shift @col;
  $psymtab{$phone} = $id;
  $idx ++;
}
close(P);
%pint2sym = (); 
foreach(keys %psymtab) {
  if ($pint2sym{$psymtab{$_}}) {print "--> ERROR: ID \"$psymtab{$_}\" duplicates\n"; exit 1;} 
  else {$pint2sym{$psymtab{$_}} = $_;}
}
print "--> $lang/phones.txt is OK\n";
print "\n";

# Check word.txt -------------------------------
print "Checking words.txt: #0 ...\n";
if (-z "$lang/words.txt") {print "--> ERROR: $lang/words.txt is empty or does not exist\n"; exit 1;}
if (!open(W, "<$lang/words.txt")) {print "--> ERROR: fail to open $lang/words.txt\n"; exit 1;}
$idx = 1;
%wsymtab = ();
while(<W>) {
  chomp;
  my @col = split(" ", $_);
  if (@col != 2) {print "--> ERROR: expect 2 columns in $lang/words.txt (line $idx)\n"; exit 1;}
  $word = shift @col;
  $id = shift @col;
  $wsymtab{$word} = $id;
  $idx ++;
}
close(W);
%wint2sym = (); 
foreach(keys %wsymtab) {
  if ($wint2sym{$wsymtab{$_}}) {print "--> ERROR: ID \"$wsymtab{$_}\" duplicates\n"; exit 1;} 
  else {$wint2sym{$wsymtab{$_}} = $_;}
}
if (exists $wsymtab{"#0"}) {
  print "--> $lang/words.txt has \"#0\"\n";
  print "--> $lang/words.txt is OK\n";
} else {
  $warning = 1;
  print "--> WARNING: $lang/words.txt doesn't have \"#0\"\n";
  print "-->          (if you are using ARPA-type language models, you will normally\n";
  print "-->           need the disambiguation symbol \"#0\" to ensure determinizability)\n";
}
print "\n";

# Checking phones/* -------------------------------
sub check_txt_int_csl {
  my ($cat, $symtab) = @_;
  print "Checking $cat.\{txt, int, csl\} ...\n";
  if (!open(TXT, "<$cat.txt")) {$exit = 1; return print "--> ERROR: fail to open $cat.txt\n";}
  if (!open(INT, "<$cat.int")) {$exit = 1; return print "--> ERROR: fail to open $cat.int\n";}
  if (!open(CSL, "<$cat.csl")) {$exit = 1; return print "--> ERROR: fail to open $cat.csl\n";}

  if (-z "$cat.txt") {$warning = 1; print "--> WARNING: $cat.txt is empty\n";}
  if (-z "$cat.int") {$warning = 1; print "--> WARNING: $cat.int is empty\n";}
  if (-z "$cat.csl") {$warning = 1; print "--> WARNING: $cat.csl is empty\n";}

  $idx1 = 1;
  while(<TXT>) {
    chomp;
    my @col = split(" ", $_);
    if (@col != 1) {$exit = 1; return print "--> ERROR: expect 1 column in $cat.txt (break at line $idx1)\n";}
    $entry[$idx1] = shift @col;
    $idx1 ++;
  }
  close(TXT); $idx1 --;
  print "--> $idx1 entry/entries in $cat.txt\n";

  $idx2 = 1;
  while(<INT>) {
    chomp;
    my @col = split(" ", $_);
    if (@col != 1) {$exit = 1; return print "--> ERROR: expect 1 column in $cat.int (break at line $idx2)\n";}
    if ($symtab->{$entry[$idx2]} ne shift @col) {$exit = 1; return print "--> ERROR: $cat.int doesn't correspond to $cat.txt (break at line $idx2)\n";}
    $idx2 ++;
  }
  close(INT); $idx2 --;
  if ($idx1 != $idx2) {$exit = 1; return print "--> ERROR: $cat.int doesn't correspond to $cat.txt (break at line ", $idx2+1, ")\n";}
  print "--> $cat.int corresponds to $cat.txt\n";

  $num_lines = 0;
  while(<CSL>) {
    chomp;
    my @col = split(":", $_);
    $num_lines++;
    if (@col != $idx1) {$exit = 1; return print "--> ERROR: expect $idx1 block/blocks in $cat.csl (break at line $idx3)\n";}
    foreach(1 .. $idx1) {
      if ($symtab->{$entry[$_]} ne @col[$_-1]) {$exit = 1; return print "--> ERROR: $cat.csl doesn't correspond to $cat.txt (break at line $idx3, block $_)\n";}
    }
  }
  close(CSL);
  if ($idx1 != 0) { # nonempty .txt,.int files
    if ($num_lines != 1) {
      $exit = 1; 
      return print "--> ERROR: expect 1 line in $cat.csl\n";
    }
  } else {
    if ($num_lines != 1 && $num_lines != 0) {
      $exit = 1;
      return print "--> ERROR: expect 0 or 1 line in $cat.csl, since empty .txt,int\n";
    }
  }
  print "--> $cat.csl corresponds to $cat.txt\n";

  return print "--> $cat.\{txt, int, csl\} are OK\n";
}

sub check_txt_int {
  my ($cat, $symtab, $sym_check) = @_;
  print "Checking $cat.\{txt, int\} ...\n";
  if (-z "$cat.txt") {$exit = 1; return print "--> ERROR: $cat.txt is empty or does not exist\n";}
  if (-z "$cat.int") {$exit = 1; return print "--> ERROR: $cat.int is empty or does not exist\n";}
  if (!open(TXT, "<$cat.txt")) {$exit = 1; return print "--> ERROR: fail to open $cat.txt\n";}
  if (!open(INT, "<$cat.int")) {$exit = 1; return print "--> ERROR: fail to open $cat.int\n";}

  $idx1 = 1;
  while(<TXT>) {
    chomp;
    s/^(shared|not-shared) (split|not-split) //g;
    s/ nonword$//g;
    s/ begin$//g;
    s/ end$//g;
    s/ internal$//g;
    s/ singleton$//g;
    $entry[$idx1] = $_;
    $idx1 ++; 
  }
  close(TXT); $idx1 --;
  print "--> $idx1 entry/entries in $cat.txt\n";

	my %used_syms = ();
  $idx2 = 1;
  while(<INT>) {
    chomp;
    s/^(shared|not-shared) (split|not-split) //g;
    s/ nonword$//g;
    s/ begin$//g;
    s/ end$//g;
    s/ internal$//g;
    s/ singleton$//g;
    my @col = split(" ", $_);
    @set = split(" ", $entry[$idx2]);
    if (@set != @col) {$exit = 1; return print "--> ERROR: $cat.int doesn't correspond to $cat.txt (break at line $idx2)\n";}
    foreach(0 .. @set-1) {
      if ($symtab->{@set[$_]} ne @col[$_]) {$exit = 1; return print "--> ERROR: $cat.int doesn't correspond to $cat.txt (break at line $idx2, block " ,$_+1, ")\n";}
			if ($sym_check && defined $used_syms{@set[$_]}) {$exit = 1; return print "--> ERROR: $cat.txt and $cat.int contain duplicate symbols (break at line $idx2, block " ,$_+1, ")\n";}
			$used_syms{@set[$_]} = 1;
    }
    $idx2 ++;
  }
  close(INT); $idx2 --;
  if ($idx1 != $idx2) {$exit = 1; return print "--> ERROR: $cat.int doesn't correspond to $cat.txt (break at line ", $idx2+1, ")\n";}
  print "--> $cat.int corresponds to $cat.txt\n";

	if ($sym_check) {
		while ( my ($key, $value) = each(%silence) ) {
			if (!defined $used_syms{$key}) {$exit = 1; return print "--> ERROR: $cat.txt and $cat.int do not contain all silence phones\n";}
		}
		while ( my ($key, $value) = each(%nonsilence) ) {
			if (!defined $used_syms{$key}) {$exit = 1; return print "--> ERROR: $cat.txt and $cat.int do not contain all non-silence phones\n";}
		}
	}

  return print "--> $cat.\{txt, int\} are OK\n";
}

# Check disjoint and summation -------------------------------
sub intersect {
  my ($a, $b) = @_;
  @itset = ();
  %itset = ();
  foreach(keys %$a) {
    if (exists $b->{$_} and !$itset{$_}) {
      push(@itset, $_);
      $itset{$_} = 1;
    }
  }
  return @itset;
}

sub check_disjoint {
  print "Checking disjoint: silence.txt, nonsilence.txt, disambig.txt ...\n";
  if (!open(S, "<$lang/phones/silence.txt"))    {$exit = 1; return print "--> ERROR: fail to open $lang/phones/silence.txt\n";}
  if (!open(N, "<$lang/phones/nonsilence.txt")) {$exit = 1; return print "--> ERROR: fail to open $lang/phones/nonsilence.txt\n";}
  if (!open(D, "<$lang/phones/disambig.txt"))   {$exit = 1; return print "--> ERROR: fail to open $lang/phones/disambig.txt\n";}

  $idx = 1;
  while(<S>) {
    chomp;
    my @col = split(" ", $_);
    $phone = shift @col;
    if ($silence{$phone}) {$exit = 1; print "--> ERROR: phone \"$phone\" duplicates in $lang/phones/silence.txt (line $idx)\n";}
    $silence{$phone} = 1;
    push(@silence, $phone);
    $idx ++;
  }
  close(S);

  $idx = 1; 
  while(<N>) {
    chomp;
    my @col = split(" ", $_);
    $phone = shift @col;
    if ($nonsilence{$phone}) {$exit = 1; print "--> ERROR: phone \"$phone\" duplicates in $lang/phones/nonsilence.txt (line $idx)\n";}
    $nonsilence{$phone} = 1;
    push(@nonsilence, $phone);
    $idx ++;
  }
  close(N);

  $idx = 1;
  while(<D>) {
    chomp;
    my @col = split(" ", $_);
    $phone = shift @col;
    if ($disambig{$phone}) {$exit = 1; print "--> ERROR: phone \"$phone\" duplicates in $lang/phones/disambig.txt (line $idx)\n";}
    $disambig{$phone} = 1;
    $idx ++;
  }
  close(D);

  my @itsect1 = intersect(\%silence, \%nonsilence);
  my @itsect2 = intersect(\%silence, \%disambig);
  my @itsect3 = intersect(\%disambig, \%nonsilence);

  $success = 1;
  if (@itsect1 != 0) {
    $success = 0;
    $exit = 1; print "--> ERROR: silence.txt and nonsilence.txt have intersection -- ";
    foreach(@itsect1) {
      print $_, " ";
    }
    print "\n";
  } else {print "--> silence.txt and nonsilence.txt are disjoint\n";}

  if (@itsect2 != 0) {
    $success = 0;
    $exit = 1; print "--> ERROR: silence.txt and disambig.txt have intersection -- ";
    foreach(@itsect2) {
      print $_, " ";
    }
    print "\n";
  } else {print "--> silence.txt and disambig.txt are disjoint\n";}

  if (@itsect3 != 0) {
    $success = 0;
    $exit = 1; print "--> ERROR: disambig.txt and nonsilence.txt have intersection -- ";
    foreach(@itsect1) {
      print $_, " ";
    }
    print "\n";
  } else {print "--> disambig.txt and nonsilence.txt are disjoint\n";}

  $success == 0 || print "--> disjoint property is OK\n";
  return;
}

sub check_summation {
  print "Checking sumation: silence.txt, nonsilence.txt, disambig.txt ...\n";
  if (scalar(keys %silence) == 0)    {$exit = 1; return print "--> ERROR: $lang/phones/silence.txt is empty or does not exist\n";}
  if (scalar(keys %nonsilence) == 0) {$exit = 1; return print "--> ERROR: $lang/phones/nonsilence.txt is empty or does not exist\n";}
  if (scalar(keys %disambig) == 0)   {$warning = 1; print "--> WARNING: $lang/phones/disambig.txt is empty or does not exist\n";}

  %sum = (%silence, %nonsilence, %disambig);
  $sum{"<eps>"} = 1;

  my @itset = intersect(\%sum, \%psymtab);
  my @key1 = keys %sum;
  my @key2 = keys %psymtab;
  my %itset = (); foreach(@itset) {$itset{$_} = 1;}
  if (@itset < @key1) {
    $exit = 1; print "--> ERROR: phones in silence.txt, nonsilence.txt, disambig.txt but not in phones.txt -- ";
    foreach(@key1) {
      if (!$itset{$_}) {print "$_ ";}
    }
    print "\n";
  }

  if (@itset < @key2) {
    $exit = 1; print "--> ERROR: phones in phones.txt but not in silence.txt, nonsilence.txt, disambig.txt -- ";
    foreach(@key2) {
      if (!$itset{$_}) {print "$_ ";}
    }
    print "\n";
  }

  if (@itset == @key1 and @itset == @key2) {
    print "--> summation property is OK\n";
  }
  return;
}

%silence = ();
@silence = ();
%nonsilence = ();
@nonsilence = ();
%disambig = ();
check_disjoint; print "\n";
check_summation; print "\n";

@list1 = ("context_indep", "disambig", "nonsilence", "silence", "optional_silence");
@list2 = ("roots", "sets");
foreach(@list1) {
  check_txt_int_csl("$lang/phones/$_", \%psymtab); print "\n";
}
foreach(@list2) {
  check_txt_int("$lang/phones/$_", \%psymtab, 1); print "\n";
}
if ((-s "$lang/phones/extra_questions.txt") || (-s "$lang/phones/extra_questions.int")) {
  check_txt_int("$lang/phones/extra_questions", \%psymtab, 0); print "\n";
} else {
  print "Checking $lang/phones/extra_questions.\{txt, int\} ...\n";
  if ((-f "$lang/phones/extra_questions.txt") && (-f "$lang/phones/extra_questions.int")) {
    print "--> WARNING: the optional $lang/phones/extra_questions.\{txt, int\} are empty!\n\n";
    $warning = 1;
  } else {
    print "--> ERROR: $lang/phones/extra_questions.\{txt, int\} do not exist (they may be empty, but should be present)\n\n";
    $exit = 1;
  }
} 
if (-e "$lang/phones/word_boundary.txt") {
  check_txt_int("$lang/phones/word_boundary", \%psymtab, 0); print "\n";
}

# Checking optional_silence.txt -------------------------------
print "Checking optional_silence.txt ...\n";
$idx = 1;
$success = 1;
if (-z "$lang/phones/optional_silence.txt") {$exit = 1; $success = 0; print "--> ERROR: $lang/phones/optional_silence.txt is empty or does not exist\n";}
if (!open(OS, "<$lang/phones/optional_silence.txt")) {$exit = 1; $success = 0; print "--> ERROR: fail to open $lang/phones/optional_silence.txt\n";}
print "--> reading $lang/phones/optional_silence.txt\n";
while(<OS>) {
  chomp;
  my @col = split(" ", $_);
  if ($idx > 1 or @col > 1) {
    $exit = 1; print "--> ERROR: only 1 phone expected in $lang/phones/optional_silence.txt\n"; $success = 0;
  } elsif (!$silence{$col[0]}) {
    $exit = 1; print "--> ERROR: phone $col[0] not found in $lang/phones/silence_phones.txt\n"; $success = 0;
  }
  $idx ++;
}
close(OS);
$success == 0 || print "--> $lang/phones/optional_silence.txt is OK\n";
print "\n";

# Check disambiguation symbols -------------------------------
print "Checking disambiguation symbols: #0 and #1\n";
if (scalar(keys %disambig) == 0) {$warning = 1; print "--> WARNING: $lang/phones/disambig.txt is empty or does not exist\n";}
if (exists $disambig{"#0"} and exists $disambig{"#1"}) {
  print "--> $lang/phones/disambig.txt has \"#0\" and \"#1\"\n";
  print "--> $lang/phones/disambig.txt is OK\n\n";
} else {
  print "--> WARNING: $lang/phones/disambig.txt doesn't have \"#0\" or \"#1\";\n";
  print "-->          this would not be OK with a conventional ARPA-type language\n";
  print "-->          model or a conventional lexicon (L.fst)\n";
  $warning = 1;
}


# Check topo -------------------------------
print "Checking topo ...\n";
if (-z "$lang/topo") {$exit = 1; print "--> ERROR: $lang/topo is empty or does not exist\n";}
if (!open(T, "<$lang/topo")) {$exit = 1; print "--> ERROR: fail to open $lang/topo\n";}
$idx = 1;
while(<T>) {
  chomp;
  next if (m/^<.*>[ ]*$/);
  if ($idx == 1) {$nonsilence_seq = $_; $idx ++;}
  if ($idx == 2) {$silence_seq = $_;}
}
close(T);
if ($silence_seq == 0 || $nonsilence_seq == 0) {$exit = 1; print "--> ERROR: $lang/topo doesn't have nonsilence section or silence section\n";}
@silence_seq = split(" ", $silence_seq);
@nonsilence_seq = split(" ", $nonsilence_seq);
$success1 = 1;
if (@nonsilence_seq != @nonsilence) {$exit = 1; print "--> ERROR: $lang/topo's nonsilence section doesn't correspond to nonsilence.txt\n";}
else {
  foreach(0 .. scalar(@nonsilence)-1) {
    if ($psymtab{@nonsilence[$_]} ne @nonsilence_seq[$_]) {
      $exit = 1; print "--> ERROR: $lang/topo's nonsilence section doesn't correspond to nonsilence.txt\n";
      $success = 0;
    }
  }
}
$success1 != 1 || print "--> $lang/topo's nonsilence section is OK\n";
$success2 = 1;
if (@silence_seq != @silence) {$exit = 1; print "--> ERROR: $lang/topo's silence section doesn't correspond to silence.txt\n";}
else {
  foreach(0 .. scalar(@silence)-1) {
    if ($psymtab{@silence[$_]} ne @silence_seq[$_]) {
      $exit = 1; print "--> ERROR: $lang/topo's silence section doesn't correspond to silence.txt\n";
      $success = 0;
    }
  }
}
$success2 != 1 || print "--> $lang/topo's silence section is OK\n";
$success1 != 1 or $success2 != 1 || print "--> $lang/topo is OK\n";
print "\n";

# Check word_boundary -------------------------------
$nonword   = "";
$begin     = "";
$end       = "";
$internal  = "";
$singleton = "";
if (-s "$lang/phones/word_boundary.txt") {
  print "Checking word_boundary.txt: silence.txt, nonsilence.txt, disambig.txt ...\n";
  if (!open (W, "<$lang/phones/word_boundary.txt")) {$exit = 1; print "--> ERROR: fail to open $lang/phones/word_boundary.txt\n";}
  $idx = 1;
  %wb = ();
  while(<W>) {
    chomp;
    my @col;
    if (m/^.*nonword$/  ) {s/ nonword//g;    @col = split(" ", $_); if (@col == 1) {$nonword   .= "$col[0] ";}}
    if (m/^.*begin$/    ) {s/ begin$//g;     @col = split(" ", $_); if (@col == 1) {$begin     .= "$col[0] ";}}
    if (m/^.*end$/      ) {s/ end$//g;       @col = split(" ", $_); if (@col == 1) {$end       .= "$col[0] ";}}
    if (m/^.*internal$/ ) {s/ internal$//g;  @col = split(" ", $_); if (@col == 1) {$internal  .= "$col[0] ";}}
    if (m/^.*singleton$/) {s/ singleton$//g; @col = split(" ", $_); if (@col == 1) {$singleton .= "$col[0] ";}}
    if (@col != 1) {$exit = 1; print "--> ERROR: expect 1 column in $lang/phones/word_boundary.txt (line $idx)\n";}
    $wb{shift @col} = 1;
    $idx ++;
  }
  close(W);

  @itset = intersect(\%disambig, \%wb);
  $success1 = 1;
  if (@itset != 0) {
    $success1 = 0;
    $exit = 1; print "--> ERROR: $lang/phones/word_boundary.txt has disambiguation symbols -- ";
    foreach(@itset) {print "$_ ";}
    print "\n";
  }
  $success1 == 0 || print "--> $lang/phones/word_boundary.txt doesn't include disambiguation symbols\n";

  %sum = (%silence, %nonsilence);
  @itset = intersect(\%sum, \%wb);
  %itset = (); foreach(@itset) {$itset{$_} = 1;}
  $success2 = 1;
  if (@itset < scalar(keys %sum)) {
    $success2 = 0;
    $exit = 1; print "--> ERROR: phones in nonsilence.txt and silence.txt but not in word_boundary.txt -- ";
    foreach(keys %sum) {
      if (!$itset{$_}) {print "$_ ";}            
    }
    print "\n";
  }
  if (@itset < scalar(keys %wb)) {
    $success2 = 0;
    $exit = 1; print "--> ERROR: phones in word_boundary.txt but not in nonsilence.txt or silence.txt -- ";
    foreach(keys %wb) {
      if (!$itset{$_}) {print "$_ ";}
    }
    print "\n";
  }
  $success2 == 0 || print "--> $lang/phones/word_boundary.txt is the union of nonsilence.txt and silence.txt\n";
  $success1 != 1 or $success2 != 1 || print "--> $lang/phones/word_boundary.txt is OK\n";
  print "\n";
}

if (-s "$lang/phones/word_boundary.int") {
  print "Checking word_boundary.int and disambig.int\n";
  if (!open (W, "<$lang/phones/word_boundary.int")) {$exit = 1; print "--> ERROR: fail to open $lang/phones/word_boundary.int\n";}
  while (<W>) {
    @A = split;
    if (@A != 2) { $exit = 1; print "--> ERROR: bad line $_ in $lang/phones/word_boundary.int\n"; }
    $wbtype{$A[0]} = $A[1];
  }
  close(W);
  if (!open (D, "<$lang/phones/disambig.int")) {$exit = 1; print "--> ERROR: fail to open $lang/phones/disambig.int\n";}
  while (<D>) { 
    @A = split;
    if (@A != 1) { $exit = 1; print "--> ERROR: bad line $_ in $lang/phones/disambig.int\n"; }
    $is_disambig{$A[0]} = 1;
  }

  foreach $fst ("L.fst", "L_disambig.fst") {
    $wlen = int(rand(100)) + 1;
    print "--> generating a $wlen words sequence\n";
    $wordseq = "";
    $sid = 0;
    foreach (1 .. $wlen) {
      $id = int(rand(scalar(%wint2sym)));
      while ($wint2sym{$id} =~ m/^#[0-9]*$/ or $id == 0) {
        $id = int(rand(scalar(%wint2sym)));
      }
      $wordseq = $wordseq . "$sid ". ($sid + 1) . " $id $id 0\n";
      $sid ++;
    }
    $wordseq = $wordseq . "$sid 0";
    $phoneseq = `echo \"$wordseq" | fstcompile | fstcompose $lang/$fst - | fstproject | fstrandgen | fstrmepsilon | fsttopsort | fstprint | awk '{if (NF > 2) {print \$3}}';`;
    @phoneseq = split(" ", $phoneseq);
    $transition = { }; # empty assoc. array of allowed transitions between phone types.  1 means we count a word,
    # 0 means transition is allowed.  bos and eos are added as extra symbols here.
    foreach $x ("bos", "nonword", "end", "singleton") {
      $transition{$x, "nonword"} = 0;
      $transition{$x, "begin"} = 1;
      $transition{$x, "singleton"} = 1;
      $transition{$x, "eos"} = 0;
    }
    $transition{"begin", "end"} = 0;
    $transition{"begin", "internal"} = 0;
    $transition{"internal", "internal"} = 0;
    $transition{"internal", "end"} = 0;

    $cur_state = "bos";
    $num_words = 0;
    foreach $phone (split (" ", "$phoneseq eos")) {
      if (!($fst == "L_disambig.fst" && defined $is_disambig{$phone})) {
        if ($phone == "eos") {
          $state = "eos";
        } else {
          $state = $wbtype{$phone};
        }
        if (!defined $state) {
          $exit = 1; print "--> ERROR: phone $phone is not specified in $lang/phones/word_boundary.int\n";
          last;
        } elsif (!defined $transition{$cur_state, $state}) {
          $exit = 1; print "--> ERROR: transition from state $cur_state to $state indicates error in word_boundary.int or L.fst\n";
          last;
        } else {
          $num_words += $transition{$cur_state, $state};
          $cur_state = $state;
        }
      }
    }
    if (!$exit) {
      if ($num_words != $wlen) {
        $exit = 1; print "--> ERROR: number of reconstructed words $num_words does not match real number of words $wlen; indicates problem in $fst or word_boundary.int.  phoneseq = $phoneseq\n";
      } else {
        print "--> resulting phone sequence from $fst corresponds to the word sequence\n";
        print "--> $fst is OK\n";
      }
    }
  }
  print "\n";
}

# Check oov -------------------------------
check_txt_int("$lang/oov", \%wsymtab, 0); print "\n";


# Check determinizability of G.fst
if (-e "$lang/G.fst") {
  $cmd = "fstdeterminize $lang/G.fst /dev/null";
  $ret = system(". ./path.sh; $cmd");
  if ($ret == 0) {
    print "--> command $cmd succeeded\n";
  } else {
    print "--> ERROR: command $cmd failed\n";
    $exit = 1;
  }
}

if (-e "$lang/G.fst" && -e "$lang/L_disambig.fst") {
  $cmd = "fstcompose $lang/L_disambig.fst $lang/G.fst | fstdeterminize > /dev/null";
  $ret = system(". ./path.sh; $cmd");
  if ($ret == 0) {
    print "--> command $cmd succeeded\n";
  } else {
    print "--> ERROR: command $cmd failed\n";
    $exit = 1;
  }
}


if ($exit == 1) { print "--> ERROR (see error messages above)\n"; exit 1;}
else {
  if ($warning == 1) { print "--> WARNING (check output above for warnings)\n"; exit 0; }
  else { print "--> SUCCESS [validating lang directory $lang]\n"; exit 0; }
}

