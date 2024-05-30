#!/usr/bin/env perl
#
# Copyright 2014  David Snyder
# Apache 2.0.
#
# Creates the target and nontarget files used by score_lre07.v01d.pl for
# NIST LRE 2007 General Language Recognition closed-set evaluation.
# See http://www.itl.nist.gov/iad/mig//tests/lre/2007/LRE07EvalPlan-v8b.pdf
# for more details on the evaluation. 

if (@ARGV != 5) {
  print STDERR "Usage: $0 <path-to-posteriors> <path-to-utt2lang> \
    <path-to-languages.txt> <path-to-targets-output> \
    <path-to-nontargets-output>\n";
  exit(1);
}

($posts, $utt2lang, $languages, $targets, $nontargets) = @ARGV;
%lang_to_idx = ();
%idx_to_lang = ();
%utt_to_lang = ();
$oos_lang = "zzz";
open(LANG2IDX, "<", $languages) || die "Cannot open $languages file";
while (<LANG2IDX>) {
  chomp;
  @toks = split(" ", $_);
  $lang = $toks[0];
  $idx = $toks[1];
  $lang_to_idx{$lang} = $idx;
  $idx_to_lang{$idx} = $lang;
}
close(LANG2IDX) || die;

open(UTT2LANG, "<", $utt2lang) || die "Cannot open $utt2lang file";
while (<UTT2LANG>) {
  chomp;
  @toks = split(" ", $_);
  $utt = $toks[0];
  $lang = $toks[1];
  $utt_to_lang{$utt} = $lang;
}
close(UTT2LANG) || die;

open(POSTS, "<", $posts) || die "Cannot open $posts file";
open(TARGETS, ">", $targets) || die "Cannot open $targets file";
open(NONTARGETS, ">", $nontargets) || die "Cannot open $nontargets file";
while($line = <POSTS>) {
  chomp($line);
  $line =~ s/[\[\]]//g;
  @toks = split(" ", $line);
  $utt = $toks[0];
  $actual_lang = $utt_to_lang{$utt};
  $size = $#toks + 1;
  $max_lang = "zzz";
  $max_log_prob = -9**9**9; #-inf
  $target_prob = 0;
  # Handle target
  for ($i = 1; $i < $size; $i++) {
    if ($max_log_prob < $toks[$i]) {
      $max_log_prob = $toks[$i];
      $max_lang = $idx_to_lang{$i-1};
    }
    if ($actual_lang eq $idx_to_lang{$i-1}) {
      print "$actual_lang $idx_to_lang{$i-1}\n";
    }
    if (index($actual_lang, $idx_to_lang{$i-1}) != -1 
      || $actual_lang eq $idx_to_lang{$i-1}) {
      $target_prob = exp($toks[$i]); 
    }
  }

  if (index($actual_lang, ".") != -1) {
    @lang_parts = split("[.]", $actual_lang);
    $lang = $lang_parts[0];
  } else {
    $lang = $actual_lang;
  }
  if ($lang =~ /(arabic|bengali|farsi|german|japanese|korean|russian|tamil|thai|vietnamese|chinese|english|hindustani|spanish)/i) {
    if (index($actual_lang, $max_lang) != -1 || $actual_lang eq $max_lang) {
      print TARGETS "general_lr $lang closed_set $utt t $target_prob "
            ."$actual_lang\n";
    } else {
      print TARGETS "general_lr $lang closed_set $utt f $target_prob "
            ."$actual_lang\n";
    }
  }
  # Handle nontarget
  for ($i = 1; $i < $size; $i++) {
    $nontarget_lang = $idx_to_lang{$i-1};
    next if (index($actual_lang, $nontarget_lang) != -1 
      || $actual_lang eq $nontarget_lang);

    # if the nontarget lang is most probable
    if ($nontarget_lang =~ /(arabic|bengali|farsi|german|japanese|korean|russian|tamil|thai|vietnamese|chinese|english|hindustani|spanish)/i) {
      $prob = exp($toks[$i]);
      if (index($max_lang, $nontarget_lang) != -1 
        || $max_lang eq $nontarget_lang) {
        print NONTARGETS "general_lr $nontarget_lang closed_set $utt t "
              ."$prob $actual_lang\n";
      } else {
        print NONTARGETS "general_lr $nontarget_lang closed_set $utt f "
              ."$prob $actual_lang\n";
      }
    }
  }
}
close(POSTS) || die;
close(TARGETS) || die;
close(NONTARGETS) || die;
