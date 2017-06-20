#!/usr/bin/env perl

use IPC::Open2;

if (@ARGV != 1) {
  print "Usage: $0 [options] <lang_directory>\n";
  print "e.g.:  $0 data/lang\n";
  exit(1);
}

$lang = shift @ARGV;

# This script checks that G.fst in the lang.fst directory is OK with respect
# to certain expected properties, and returns nonzero exit status if a problem was
# detected.  It is called from validate_lang.pl.
# This only checks the properties of G that relate to disambiguation symbols,
# epsilons and forbidden symbols <s> and </s>.

if (! -e "$lang/G.fst") {
  print "$0: error: $lang/G.fst does not exist\n";
  exit(1);
}

open(W, "<$lang/words.txt") || die "opening $lang/words.txt";
$hash_zero = -1;
while (<W>) {
  @A = split(" ", $_);
  ($sym, $int) = @A;
  if ($sym eq "<s>" || $sym eq "</s>") { $is_forbidden{$int} = 1; }
  if ($sym eq "#0") { $hash_zero = $int; }
}

if (-e "$lang/phones/wdisambig_words.int") {
  open(F, "<$lang/phones/wdisambig_words.int") || die "opening $lang/phones/wdisambig_words.int";
  while (<F>) {
    chop;
    $is_disambig{$_} = 1;
  }
} else {
  $is_disambig{$hash_zero} = 1;
}

$input_cmd = ". ./path.sh; fstprint $lang/G.fst|";
open(G, $input_cmd) || die "running command $input_cmd";

$info_cmd = ". ./path.sh; fstcompile | fstinfo ";
open2(O, I, "$info_cmd") || die "running command $info_cmd";

$has_epsilons = 0;

while (<G>) {
  @A = split(" ", $_);
  if (@A >= 4) {
    if ($is_forbidden{$A[2]} || $is_forbidden{$A[3]}) {
      chop;
      print "$0: validating $lang: error: line $_ in G.fst contains forbidden symbol <s> or </s>\n";
      exit(1);
    } elsif ($is_disambig{$A[2]}) {
      print I $_;
      if ($A[3] != 0) {
        chop;
        print "$0: validating $lang: error: line $_ in G.fst has disambig on input but no epsilon on output\n";
        exit(1);
      }
    } elsif ($A[2] == 0) {
      print I $_;
      $has_epsilons = 1;
    } elsif ($A[2] != $A[3]) {
      chop;
      print "$0: validating $lang: error: line $_ in G.fst has inputs and outputs different but input is not disambig symbol.\n";
      exit(1);
    }
  }
}

close(I);  # tell 'fstcompile | fstinfo' pipeline that its input is done.
while (<O>) {
  if (m/cyclic\s+y/) {
    print "$0: validating $lang: error: G.fst has cycles containing only disambig symbols and epsilons.  Would cause determinization failure\n";
    exit(1);
  }
}

if ($has_epsilons) {
  print "$0: warning: validating $lang: G.fst has epsilon-input arcs.  We don't expect these in most setups.\n";
}

print "--> $0 successfully validated $lang/G.fst\n";
exit(0);
