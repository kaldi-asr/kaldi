#!/usr/bin/env perl
# Removes the dialect parts on an utt2lang file.
# For example <utt> chinese.wu is converted to <utt> chinese.

my ($utt2lang_file) = @ARGV;
open(UTT2LANG, "<$utt2lang_file") or die "no utt2lang file";
$utt2lang_short = "";
while(<UTT2LANG>) {
  $line = $_;
  chomp($line);
  @words = split(" ", $line);
  $utt = $words[0];
  $lang_long = $words[1];
  @lang_parts = split('[.]', $lang_long);
  # The actual language. Other parts are dialects or subcategories.
  $lang = $lang_parts[0];
  $utt2lang_short .= $utt . " " . $lang . "\n";
}
print $utt2lang_short;
