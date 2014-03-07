#!/usr/bin/perl -w

my ($train_file, $test_file, $lang_file, $priors_file) = @ARGV;
open(UTT2LANG_TRAIN, "<$train_file") or die "no utt2lang training file";

%train_count = ();
$train_tot = 0;
while(<UTT2LANG_TRAIN>) {
  $line = $_;
  chomp($line);
  @words = split(" ", $line);
  $lang = $words[1];
  if (not exists($train_count{$lang})) {
    $train_count{$lang} = 1;
  } else {
    $train_count{$lang} += 1;
  }
  $train_tot += 1;
}

open(UTT2LANG_TEST, "<$test_file");

%test_count = ();
$test_tot = 0;
while(<UTT2LANG_TEST>) {
  $line = $_;
  chomp($line);
  @words = split(" ", $line);
  $lang = $words[1];
  if (not exists($test_count{$lang})) {
    $test_count{$lang} = 1;
  } else {
    $test_count{$lang} += 1;
  }
  $test_tot += 1;
}

foreach my $key (keys %train_count) {
  if (not exists($test_count{$key})) {
    $test_count{$key} = 0;
  }
}

# load languages file
open(LANGUAGES, "<$lang_file");
@idx_to_lang = ();

$largest_idx = 0;
while(<LANGUAGES>) {
  $line = $_;
  chomp($line);
  @words = split(" ", $line);
  $lang = $words[0];
  $idx = $words[1];
  $idx_to_lang[$idx + 0] = $lang;
  if ($idx > $largest_idx) {
    $largest_idx = $idx;
  }
}

$priors = " [ ";
foreach $lang (@idx_to_lang) {
  $ratio = (1.0*$test_count{$lang}) / $train_count{$lang};
  $priors .= "$ratio ";
}

$priors .= " ]";
open(PRIORS, ">$priors_file");
print PRIORS $priors;
