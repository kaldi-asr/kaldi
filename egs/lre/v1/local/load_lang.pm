#!/usr/bin/perl

sub load_lang {
  $lang_abbreviation_file = $_[0];
  open(LANG, "<", $lang_abbreviation_file) or die "cannot open file $lang_abbreviation_file";
  $num=0;
  while(<LANG>) {
    chomp;
    $line = $_;
    @l=split("[ ]",$line);
    $long_lang{$l[0]} = $l[1];
    $abbr_lang{$l[1]} = $l[0];
    $num_lang{$l[0]} = sprintf("%04d", $num);
    $num++;
  }
  close LANG;
  return (%long_lang, %abbr_lang, %num_lang);
}
1;
